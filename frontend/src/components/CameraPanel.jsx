import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  VideoOff,
  RefreshCw,
  AlertOctagon,
  AlertTriangle,
  Loader2,
  CheckCircle,
  Eye,
  Activity,
  Layers,
} from 'lucide-react';
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from '@mediapipe/tasks-vision';
import { sendPosePrediction } from '../config/api';

// Explicit, plain asset URLs pinned to installed package version @1.1.0
const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm';
const POSE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/**
 * CameraPanel
 *
 * Manages live webcam capture, permission handling, MediaPipe Pose Landmarker
 * real-time skeleton overlay, and streaming 33 pose landmarks to the FastAPI
 * backend Random Forest fall-detection model.
 */
export default function CameraPanel({
  onPredictionUpdate,
  onPredictionError,
  onCameraStatusChange,
}) {
  // State machine: 'OFF' | 'REQUESTING_PERMISSION' | 'ACTIVE' | 'PERMISSION_DENIED' | 'ERROR' | 'NOT_SUPPORTED'
  const [cameraState, setCameraState] = useState('OFF');
  const [errorMessage, setErrorMessage] = useState('');
  const [modelStatus, setModelStatus] = useState('UNINITIALIZED'); // 'UNINITIALIZED' | 'LOADING' | 'READY' | 'ERROR'
  const [modelError, setModelError] = useState('');
  const [fps, setFps] = useState(0);
  const [detectedLandmarksCount, setDetectedLandmarksCount] = useState(0);
  const [videoResolution, setVideoResolution] = useState({ width: 0, height: 0 });
  const [deviceLabel, setDeviceLabel] = useState('');

  // Real-time backend prediction state
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState(null);

  // DOM and stream references
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const initPromiseRef = useRef(null);
  const isRunningRef = useRef(false);

  // Stable callback refs to decouple camera lifecycle from parent re-renders
  const onPredictionUpdateRef = useRef(onPredictionUpdate);
  const onPredictionErrorRef = useRef(onPredictionError);
  const onCameraStatusChangeRef = useRef(onCameraStatusChange);

  useEffect(() => {
    onPredictionUpdateRef.current = onPredictionUpdate;
    onPredictionErrorRef.current = onPredictionError;
    onCameraStatusChangeRef.current = onCameraStatusChange;
  }, [onPredictionUpdate, onPredictionError, onCameraStatusChange]);

  // Throttled prediction tracking
  const lastPredictionTimeRef = useRef(0);
  const isRequestInFlightRef = useRef(false);
  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);
  const frameCounterRef = useRef(0);

  // Performance tracking
  const lastVideoTimeRef = useRef(-1);
  const lastFpsUpdateRef = useRef(performance.now());
  const frameCountRef = useRef(0);

  /**
   * Initializes MediaPipe Pose Landmarker (cached & deduplicated across start/stop)
   */
  const initPoseLandmarker = useCallback(async () => {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    if (initPromiseRef.current) return initPromiseRef.current;

    initPromiseRef.current = (async () => {
      setModelStatus('LOADING');
      setModelError('');
      console.log(`[MediaPipe] Initializing Pose Landmarker WASM binaries from: ${MEDIAPIPE_WASM_URL}`);

      let vision;
      try {
        vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
      } catch (wasmErr) {
        const msg = `Failed to load MediaPipe WASM runtime: ${wasmErr?.message || wasmErr}`;
        console.error('[MediaPipe]', msg);
        setModelStatus('ERROR');
        setModelError(msg);
        initPromiseRef.current = null;
        return null;
      }

      let landmarker = null;
      let gpuError = null;

      // 1. Attempt GPU acceleration first
      try {
        console.log('[MediaPipe] Initializing Pose Landmarker with GPU delegate...');
        landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: POSE_LANDMARKER_MODEL_URL,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        console.log('[MediaPipe] GPU initialization succeeded.');
      } catch (err) {
        gpuError = err;
        console.warn('[MediaPipe] GPU delegate failed, attempting CPU fallback:', err?.message || err);
      }

      // 2. If GPU initialization failed, attempt CPU fallback with identical vision runtime and model
      if (!landmarker) {
        try {
          console.log('[MediaPipe] Initializing Pose Landmarker with CPU delegate...');
          landmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: POSE_LANDMARKER_MODEL_URL,
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          console.log('[MediaPipe] CPU fallback initialization succeeded.');
        } catch (cpuErr) {
          const combinedMsg = `MediaPipe failed on GPU (${gpuError?.message || 'unknown error'}) and CPU fallback (${cpuErr?.message || 'unknown error'})`;
          console.error('[MediaPipe]', combinedMsg);
          setModelStatus('ERROR');
          setModelError(combinedMsg);
          initPromiseRef.current = null;
          return null;
        }
      }

      poseLandmarkerRef.current = landmarker;
      setModelStatus('READY');
      setModelError('');
      console.log('[MediaPipe] Pose Landmarker initialized successfully.');
      return landmarker;
    })();

    return initPromiseRef.current;
  }, []);

  /**
   * Animation & pose inference loop
   * Runs only when video has non-zero dimensions and is in readyState >= 2
   */
  const renderFrameLoop = useCallback(() => {
    if (!isRunningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Verify video element readiness and non-zero dimensions
    if (video && canvas && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      // Sync canvas dimensions with actual video resolution
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        setVideoResolution({ width: video.videoWidth, height: video.videoHeight });
      }

      const canvasCtx = canvas.getContext('2d');
      if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

        // Run MediaPipe inference if model is ready and frame time advanced
        if (poseLandmarkerRef.current && video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const startTimeMs = performance.now();

          try {
            const poseResults = poseLandmarkerRef.current.detectForVideo(video, startTimeMs);

            if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
              const landmarks = poseResults.landmarks[0];
              setDetectedLandmarksCount(landmarks.length);

              // Draw skeleton overlay (emerald connectors and cyan keypoints)
              const drawingUtils = new DrawingUtils(canvasCtx);
              drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
                color: '#34d399',
                lineWidth: 3,
              });
              drawingUtils.drawLandmarks(landmarks, {
                color: '#38bdf8',
                fillColor: '#0284c7',
                lineWidth: 1.5,
                radius: 4,
              });

              // Throttled prediction streaming to FastAPI backend (target ~650ms, within 500-1000ms range)
              const nowMs = performance.now();
              if (
                landmarks.length === 33 &&
                nowMs - lastPredictionTimeRef.current >= 650 &&
                !isRequestInFlightRef.current
              ) {
                lastPredictionTimeRef.current = nowMs;
                isRequestInFlightRef.current = true;
                requestIdRef.current += 1;
                const currentRequestId = requestIdRef.current;
                frameCounterRef.current += 1;
                const currentFrameId = frameCounterRef.current;

                if (abortControllerRef.current) {
                  abortControllerRef.current.abort();
                }
                const controller = new AbortController();
                abortControllerRef.current = controller;

                sendPosePrediction(landmarks, currentFrameId, 'patient-001', controller.signal)
                  .then((data) => {
                    if (isRunningRef.current && currentRequestId === requestIdRef.current) {
                      setLatestPrediction(data);
                      setPredictionError(null);
                      onPredictionUpdateRef.current?.(data);
                    }
                  })
                  .catch((err) => {
                    if (err.name === 'AbortError') return;
                    console.warn('[Prediction] API request error:', err.message);
                    if (isRunningRef.current && currentRequestId === requestIdRef.current) {
                      setPredictionError(err.message);
                      onPredictionErrorRef.current?.(err.message);
                    }
                  })
                  .finally(() => {
                    if (currentRequestId === requestIdRef.current) {
                      isRequestInFlightRef.current = false;
                    }
                  });
              }
            } else {
              setDetectedLandmarksCount(0);
            }
          } catch (inferErr) {
            console.error('[MediaPipe] Frame inference error:', inferErr);
          }
        }
      }

      // Rolling FPS calculation
      frameCountRef.current++;
      const now = performance.now();
      const elapsed = now - lastFpsUpdateRef.current;
      if (elapsed >= 500) {
        setFps(Math.round((frameCountRef.current * 1000) / elapsed));
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }
    }

    if (isRunningRef.current) {
      animFrameIdRef.current = requestAnimationFrame(renderFrameLoop);
    }
  }, []);

  /**
   * Releases hardware streams, pauses video, cancels render loop, and aborts in-flight network requests.
   * Does NOT overwrite UI state machines so errors remain displayed when stream acquisition fails.
   */
  const releaseCameraResources = useCallback(() => {
    isRunningRef.current = false;

    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    isRequestInFlightRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
        console.log(`[Webcam] Stopped camera track: "${track.label}"`);
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.onloadedmetadata = null;
      videoRef.current.oncanplay = null;
      videoRef.current.onplaying = null;
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setFps(0);
    setDetectedLandmarksCount(0);
    setDeviceLabel('');
    setLatestPrediction(null);
    setPredictionError(null);
    lastVideoTimeRef.current = -1;
  }, []);

  /**
   * Stops camera stream on explicit user action and resets UI state to OFF
   */
  const stopCamera = useCallback(() => {
    console.log('[Webcam] User initiated Stop Camera.');
    releaseCameraResources();
    setCameraState('OFF');
    onPredictionUpdateRef.current?.(null);
    onCameraStatusChangeRef.current?.('OFF');
    console.log('[Webcam] Camera stopped and resources released.');
  }, [releaseCameraResources]);

  /**
   * Starts camera on explicit user button click
   */
  const startCamera = async () => {
    console.log('[Webcam] Start Camera button clicked. Requesting camera access...');
    setErrorMessage('');

    // 1. Verify browser mediaDevices support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const unsupportedMsg =
        'Camera API is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari over HTTPS or localhost.';
      console.error('[Webcam]', unsupportedMsg);
      setCameraState('NOT_SUPPORTED');
      setErrorMessage(unsupportedMsg);
      return;
    }

    setCameraState('REQUESTING_PERMISSION');

    try {
      // Pre-initialize MediaPipe vision runtime in parallel
      initPoseLandmarker().catch((e) => console.warn('[MediaPipe] Preload warning:', e));

      // 2. Explicitly prefer built-in front-facing camera with ideal resolution
      const constraints = {
        video: {
          facingMode: 'user',
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
        },
        audio: false,
      };

      console.log('[Webcam] Requesting getUserMedia with constraints:', constraints);
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = mediaStream;

      // 3. Log selected device and track settings for debugging
      const videoTrack = mediaStream.getVideoTracks()[0];
      const selectedLabel = videoTrack?.label || 'Front-Facing Camera';
      const trackSettings = videoTrack?.getSettings ? videoTrack.getSettings() : {};
      setDeviceLabel(selectedLabel);
      console.log(`[Webcam] getUserMedia granted! Selected device: "${selectedLabel}"`);
      console.log('[Webcam] Video track settings:', trackSettings);

      // 4. Assign MediaStream to the video element
      const video = videoRef.current;
      if (!video) {
        throw new Error('Video element reference is not mounted in DOM.');
      }

      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = mediaStream;

      // 5. Activate video stream and start pose estimation render loop
      const activateStream = () => {
        if (!streamRef.current) return;
        console.log(
          `[Webcam] Video ready: ${video.videoWidth}x${video.videoHeight}, readyState: ${video.readyState}`
        );
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          setVideoResolution({ width: video.videoWidth, height: video.videoHeight });
        }
        setCameraState('ACTIVE');
        isRunningRef.current = true;
        onCameraStatusChangeRef.current?.('ACTIVE');
        if (!animFrameIdRef.current) {
          animFrameIdRef.current = requestAnimationFrame(renderFrameLoop);
        }
      };

      video.onloadedmetadata = activateStream;
      video.oncanplay = activateStream;
      video.onplaying = activateStream;

      // Start playback directly from user gesture
      try {
        await video.play();
        console.log(
          `[Webcam] video.play() resolved. Dimensions: ${video.videoWidth}x${video.videoHeight}, readyState: ${video.readyState}`
        );
      } catch (playErr) {
        console.warn('[Webcam] Immediate video.play() returned promise warning, awaiting metadata:', playErr);
      }

      if (video.readyState >= 2 || video.videoWidth > 0) {
        activateStream();
      }
    } catch (err) {
      console.error('[Webcam] getUserMedia or video initialization error:', err);
      releaseCameraResources();
      onPredictionUpdateRef.current?.(null);
      onCameraStatusChangeRef.current?.('OFF');

      let userErrorMsg = err?.message || 'An unexpected error occurred while accessing the webcam.';
      let nextState = 'ERROR';

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        nextState = 'PERMISSION_DENIED';
        userErrorMsg =
          'Webcam permission was denied. Please click the camera/lock icon in your browser address bar to allow camera access, then retry.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        nextState = 'ERROR';
        userErrorMsg = 'No camera device was detected on your system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        nextState = 'ERROR';
        userErrorMsg =
          'Camera is currently in use by another application (e.g. FaceTime, Zoom, or another browser window).';
      } else if (err.name === 'OverconstrainedError') {
        nextState = 'ERROR';
        userErrorMsg =
          'The requested camera resolution or facingMode is not supported by your camera hardware.';
      }

      setCameraState(nextState);
      setErrorMessage(userErrorMsg);
    }
  };

  /**
   * Component cleanup ONLY on unmount
   */
  useEffect(() => {
    return () => {
      console.log('[CameraPanel] Component unmounting, cleaning up all camera & MediaPipe resources...');
      releaseCameraResources();
      onPredictionUpdateRef.current?.(null);
      onCameraStatusChangeRef.current?.('OFF');

      if (poseLandmarkerRef.current) {
        try {
          poseLandmarkerRef.current.close();
        } catch (closeErr) {
          console.warn('[MediaPipe] Error closing PoseLandmarker:', closeErr);
        }
        poseLandmarkerRef.current = null;
      }
      initPromiseRef.current = null;
    };
  }, [releaseCameraResources]);

  return (
    <div className="card-panel" role="region" aria-label="Camera and Vision Feed Panel">
      {/* Panel Title Bar */}
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <Camera size={19} className="metric-icon" aria-hidden="true" />
          <span>Live Vision &amp; Pose Estimation</span>
        </h2>

        <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
          {cameraState === 'ACTIVE' ? (
            <button
              type="button"
              className="btn btn-danger"
              onClick={stopCamera}
              aria-label="Stop webcam stream"
            >
              <VideoOff size={15} aria-hidden="true" />
              <span>Stop Camera</span>
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              onClick={startCamera}
              disabled={cameraState === 'REQUESTING_PERMISSION'}
              aria-label="Start webcam stream"
            >
              {cameraState === 'REQUESTING_PERMISSION' ? (
                <>
                  <Loader2 size={15} className="spin-icon" aria-hidden="true" />
                  <span>Requesting Access...</span>
                </>
              ) : (
                <>
                  <Camera size={15} aria-hidden="true" />
                  <span>Start Camera</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport Container */}
      <div style={{ position: 'relative' }}>
        {/*
          Always keep the video and canvas elements mounted in the DOM.
          This ensures videoRef.current is never null when getUserMedia returns.
        */}
        <div
          className="camera-live-wrapper"
          style={
            cameraState === 'ACTIVE'
              ? { display: 'flex' }
              : {
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  opacity: 0,
                  pointerEvents: 'none',
                  zIndex: -1,
                }
          }
        >
          {/* Live Video Element */}
          <video
            ref={videoRef}
            playsInline
            muted
            autoPlay
            className="camera-video"
            aria-label="Live webcam feed"
          />

          {/* Skeleton Canvas Overlay directly above the video */}
          <canvas
            ref={canvasRef}
            className="camera-canvas"
            aria-label="Pose estimation skeleton overlay"
          />

          {/* Floating Live Telemetry HUD Bar */}
          <div className="camera-live-bar">
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="camera-live-badge">
                <span className="pulse-dot" style={{ backgroundColor: '#ef4444' }} aria-hidden="true" />
                <strong style={{ color: '#f87171' }}>LIVE WEBCAM</strong>
              </span>

              <span className="camera-live-badge">
                <Activity size={12} color="var(--accent-cyan)" aria-hidden="true" />
                <span>{fps} FPS</span>
              </span>

              {videoResolution.width > 0 && (
                <span className="camera-live-badge">
                  <span>
                    {videoResolution.width} &times; {videoResolution.height}
                  </span>
                </span>
              )}
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {deviceLabel && (
                <span className="camera-live-badge" title="Selected video input device">
                  <span>{deviceLabel}</span>
                </span>
              )}
              <span className="camera-live-badge">
                <Layers
                  size={12}
                  color={modelStatus === 'ERROR' ? 'var(--status-danger-text)' : 'var(--status-safe-text)'}
                  aria-hidden="true"
                />
                <span>
                  {detectedLandmarksCount > 0
                    ? `${detectedLandmarksCount} Pose Landmarks`
                    : modelStatus === 'READY'
                    ? 'Searching for subject...'
                    : modelStatus === 'ERROR'
                    ? 'Pose model failed to load'
                    : 'Loading pose model...'}
                </span>
              </span>

              {/* Live FastAPI Fall Classification Verdict */}
              {latestPrediction ? (
                <span
                  className="camera-live-badge"
                  style={{
                    backgroundColor: latestPrediction.fall_detected
                      ? 'rgba(239, 68, 68, 0.95)'
                      : 'rgba(16, 185, 129, 0.95)',
                    color: '#ffffff',
                    fontWeight: 600,
                    border: latestPrediction.fall_detected
                      ? '1px solid rgba(248, 113, 113, 0.8)'
                      : '1px solid rgba(52, 211, 153, 0.8)',
                  }}
                >
                  <Activity size={12} aria-hidden="true" />
                  <span>
                    {latestPrediction.fall_detected
                      ? `FALL DETECTED (${Math.round((latestPrediction.confidence ?? 0) * 100)}%)`
                      : `SAFE / NORMAL (${Math.round((latestPrediction.confidence ?? 0) * 100)}%)`}
                  </span>
                </span>
              ) : predictionError ? (
                <span
                  className="camera-live-badge"
                  style={{
                    backgroundColor: 'rgba(245, 158, 11, 0.95)',
                    color: '#ffffff',
                    border: '1px solid rgba(251, 191, 36, 0.8)',
                  }}
                  title={predictionError}
                >
                  <AlertTriangle size={12} aria-hidden="true" />
                  <span>FastAPI Standby</span>
                </span>
              ) : null}
            </div>
          </div>
        </div>

        {/* Inactive / Error Viewport States (Rendered when camera is not ACTIVE) */}
        {cameraState !== 'ACTIVE' && (
          <div className="camera-viewport">
            <div className="camera-grid-overlay" aria-hidden="true" />

            {cameraState === 'REQUESTING_PERMISSION' && (
              <>
                <div className="camera-placeholder-icon">
                  <Loader2 size={32} className="spin-icon" />
                </div>
                <h3 className="camera-placeholder-title">Requesting Camera Permission</h3>
                <p className="camera-placeholder-desc">
                  Please click <strong>&quot;Allow&quot;</strong> in your browser prompt to enable live pose tracking.
                </p>
              </>
            )}

            {cameraState === 'PERMISSION_DENIED' && (
              <>
                <div className="camera-placeholder-icon" style={{ color: 'var(--status-danger-text)' }}>
                  <AlertOctagon size={32} />
                </div>
                <h3 className="camera-placeholder-title" style={{ color: 'var(--status-danger-text)' }}>
                  Camera Permission Denied
                </h3>
                <div className="camera-status-callout danger">
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>How to allow camera access:</div>
                  <p style={{ fontSize: '0.82rem' }}>{errorMessage}</p>
                </div>
                <button type="button" className="btn btn-primary" onClick={startCamera}>
                  <RefreshCw size={14} aria-hidden="true" />
                  <span>Retry Permission</span>
                </button>
              </>
            )}

            {cameraState === 'ERROR' && (
              <>
                <div className="camera-placeholder-icon" style={{ color: 'var(--status-warn-text)' }}>
                  <AlertTriangle size={32} />
                </div>
                <h3 className="camera-placeholder-title" style={{ color: 'var(--status-warn-text)' }}>
                  Camera Unavailable
                </h3>
                <div className="camera-status-callout warning">
                  <p style={{ fontSize: '0.82rem' }}>{errorMessage}</p>
                </div>
                <button type="button" className="btn btn-primary" onClick={startCamera}>
                  <RefreshCw size={14} aria-hidden="true" />
                  <span>Retry Camera</span>
                </button>
              </>
            )}

            {cameraState === 'NOT_SUPPORTED' && (
              <>
                <div className="camera-placeholder-icon" style={{ color: 'var(--status-danger-text)' }}>
                  <AlertTriangle size={32} />
                </div>
                <h3 className="camera-placeholder-title">Browser Not Supported</h3>
                <div className="camera-status-callout danger">
                  <p style={{ fontSize: '0.82rem' }}>{errorMessage}</p>
                </div>
              </>
            )}

            {cameraState === 'OFF' && (
              <>
                <div className="camera-placeholder-icon">
                  <VideoOff size={32} />
                </div>
                <h3 className="camera-placeholder-title">Webcam Stream Inactive</h3>
                <p className="camera-placeholder-desc">
                  Click <strong>&quot;Start Camera&quot;</strong> to activate your front-facing camera and initialize
                  real-time MediaPipe 33-landmark pose estimation.
                </p>

                <div className="camera-integration-checklist">
                  <span className="camera-chip camera-chip-active">
                    <CheckCircle size={12} style={{ display: 'inline', marginRight: 4 }} />
                    facingMode: &quot;user&quot;
                  </span>
                  <span className="camera-chip camera-chip-active">
                    <CheckCircle size={12} style={{ display: 'inline', marginRight: 4 }} />
                    MediaPipe Tasks Pose (33 points)
                  </span>
                  <span className="camera-chip">Privacy: Stream processed strictly in-browser</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Model Error Alert Banner (if MediaPipe fails to load) */}
      {modelStatus === 'ERROR' && (
        <div
          style={{
            marginTop: '0.75rem',
            padding: '0.65rem 0.95rem',
            backgroundColor: 'var(--status-danger-bg)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--status-danger-text)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.5rem',
            fontSize: '0.78rem',
          }}
          role="alert"
        >
          <AlertOctagon size={16} style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
          <div>
            <strong>MediaPipe Initialization Error:</strong> {modelError || 'Failed to initialize Pose Landmarker.'}
          </div>
        </div>
      )}

      {/* Model & Classification Status Banner */}
      <div
        style={{
          marginTop: '1rem',
          padding: '0.75rem 1rem',
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          fontSize: '0.78rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Eye size={15} color="var(--accent-cyan)" aria-hidden="true" />
          <span>
            <strong>AI Pose Status:</strong>{' '}
            {cameraState === 'ACTIVE'
              ? modelStatus === 'READY'
                ? `Active tracking (${detectedLandmarksCount} landmarks detected)`
                : modelStatus === 'ERROR'
                ? 'Model initialization failed (see error banner above)'
                : 'Loading MediaPipe model...'
              : 'Standby (Click "Start Camera" above)'}
          </span>
        </div>

        <div style={{ color: 'var(--text-muted)' }}>
          {cameraState === 'ACTIVE' && latestPrediction ? (
            <span>
              <strong style={{ color: latestPrediction.fall_detected ? '#f87171' : '#34d399' }}>
                {latestPrediction.fall_detected ? 'Fall Alert Active' : 'Normal Posture'}
              </strong>
              {' '}&bull; FastAPI RF Model &bull; Confidence:{' '}
              <strong>{Math.round((latestPrediction.confidence ?? 0) * 100)}%</strong>
              {' '}&bull; Fall Prob:{' '}
              <strong>{Math.round((latestPrediction.fall_probability ?? 0) * 100)}%</strong>
              {' '}&bull; 47 Features on Server
            </span>
          ) : cameraState === 'ACTIVE' && predictionError ? (
            <span style={{ color: '#fbbf24' }}>
              FastAPI standby ({predictionError}). Ensure backend is running at http://localhost:8000.
            </span>
          ) : (
            <em>Trained Random Forest model loaded on FastAPI. Start camera to run real-time inference.</em>
          )}
        </div>
      </div>
    </div>
  );
}
