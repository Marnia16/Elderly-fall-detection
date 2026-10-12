import React, { useState } from 'react';
import { Camera, VideoOff, Cpu, Layers, Maximize2, RefreshCw } from 'lucide-react';
import { MOCK_SYSTEM_METRICS } from '../mock/mockData';

export default function CameraPanel() {
  const [isSimulatingStream, setIsSimulatingStream] = useState(false);

  return (
    <div className="card-panel" role="region" aria-label="Camera and Vision Feed Panel">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <Camera size={19} className="metric-icon" aria-hidden="true" />
          <span>Vision Input & Pose Estimation</span>
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span
            className="badge badge-mock"
            title="MediaPipe & Webcam integration will attach here"
          >
            [Placeholder]
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setIsSimulatingStream(!isSimulatingStream)}
            title="Toggle mock standby/preview overlay"
            style={{ fontSize: '0.74rem', padding: '0.3rem 0.6rem' }}
          >
            <RefreshCw size={12} aria-hidden="true" />
            {isSimulatingStream ? 'Show Standby State' : 'Show Grid Preview'}
          </button>
        </div>
      </div>

      <div className="camera-viewport">
        <div className="camera-grid-overlay" aria-hidden="true"></div>

        <div className="camera-placeholder-icon">
          {isSimulatingStream ? <Camera size={30} /> : <VideoOff size={30} />}
        </div>

        <h3 className="camera-placeholder-title">
          {isSimulatingStream ? 'Vision Stream Viewport Ready' : 'Webcam Integration Slot'}
        </h3>

        <p className="camera-placeholder-desc">
          This panel is the designated mounting surface for the live camera stream and
          <strong> MediaPipe Pose</strong> skeleton overlay (`&lt;video&gt;` &amp; `&lt;canvas&gt;`).
          Real-time landmark coordinates will be streamed to the FastAPI backend.
        </p>

        <div className="camera-integration-checklist">
          <span className="camera-chip camera-chip-active">
            <Cpu size={12} style={{ display: 'inline', marginRight: 4 }} />
            MediaPipe Pose: Ready to mount
          </span>
          <span className="camera-chip camera-chip-active">
            <Layers size={12} style={{ display: 'inline', marginRight: 4 }} />
            33 Landmark Canvas: Pre-wired
          </span>
          <span className="camera-chip">
            Resolution: {MOCK_SYSTEM_METRICS.camera.resolution}
          </span>
          <span className="camera-chip">
            Inference: Client-side extraction &rarr; FastAPI
          </span>
        </div>
      </div>

      <div
        style={{
          marginTop: '1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <span>
          <strong>Target Device:</strong> Local HD Webcam (AVFoundation on macOS / DirectShow on Windows)
        </span>
        <span style={{ color: 'var(--text-muted)' }}>
          FPS: <strong>0 (Camera Inactive in Foundation)</strong>
        </span>
      </div>
    </div>
  );
}
