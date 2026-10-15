import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import MetricCard from './components/MetricCard';
import CameraPanel from './components/CameraPanel';
import FallStatusCard from './components/FallStatusCard';
import IncidentLog from './components/IncidentLog';
import AlertHistory from './components/AlertHistory';
import SystemInfo from './components/SystemInfo';
import {
  Camera,
  Activity,
  UserCheck,
  Cpu,
  AlertTriangle,
} from 'lucide-react';
import {
  MOCK_DISCLAIMER_TEXT,
  MOCK_SYSTEM_METRICS,
  MOCK_ROOM_INFO,
} from './mock/mockData';

export default function App() {
  const [livePrediction, setLivePrediction] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);

  const handlePredictionUpdate = useCallback((prediction) => {
    setLivePrediction(prediction);
  }, []);

  const handlePredictionError = useCallback((error) => {
    setApiError(error);
  }, []);

  const handleCameraStatusChange = useCallback((status) => {
    setCameraActive(status === 'ACTIVE');
  }, []);

  return (
    <div className="app-container">
      {/* Top Navigation & Live Telemetry Header */}
      <Header />

      {/* Prominent Demonstration & Simulation Disclaimer */}
      <div className="disclaimer-banner" role="status">
        <div className="disclaimer-content">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>
            {cameraActive && livePrediction
              ? 'Live Inference Active: Streaming 33 MediaPipe pose landmarks to FastAPI Random Forest model.'
              : MOCK_DISCLAIMER_TEXT}
          </span>
        </div>
        <span style={{ fontSize: '0.72rem', opacity: 0.85 }}>
          {cameraActive && livePrediction
            ? '[Live Prediction Mode]'
            : '[Demonstration Foundation Mode]'}
        </span>
      </div>

      {/* Main Dashboard Layout */}
      <main className="main-content">
        {/* Metric Overview Grid */}
        <section className="metrics-grid" aria-label="System Metrics Overview">
          <MetricCard
            title="Camera Input"
            value={cameraActive ? 'Active Stream' : 'Camera Off'}
            subtitle={cameraActive ? 'Live Webcam (Front-Facing)' : 'Click Start Camera below'}
            icon={Camera}
            badge={cameraActive ? 'Live' : 'Standby'}
          />

          <MetricCard
            title="Current Fall State"
            value={
              cameraActive && livePrediction
                ? livePrediction.fall_detected
                  ? 'FALL DETECTED'
                  : 'Safe / Upright'
                : cameraActive
                ? 'Tracking Pose...'
                : 'Safe / Upright'
            }
            subtitle={
              cameraActive && livePrediction
                ? livePrediction.fall_detected
                  ? `Fall prob: ${Math.round((livePrediction.fall_probability ?? 0) * 100)}%`
                  : `Posture conf: ${Math.round((livePrediction.confidence ?? 0) * 100)}%`
                : cameraActive
                ? 'Streaming landmarks to FastAPI'
                : 'FastAPI RF Model Ready'
            }
            icon={Activity}
            badge={
              cameraActive && livePrediction
                ? livePrediction.fall_detected
                  ? 'CRITICAL ALERT'
                  : 'Live ML Safe'
                : 'Model Ready'
            }
          />

          <MetricCard
            title="Monitored Subject"
            value={MOCK_ROOM_INFO.roomNumber}
            subtitle={`Assigned: ${MOCK_ROOM_INFO.residentName}`}
            icon={UserCheck}
            badge="Room 204"
          />

          <MetricCard
            title="ML Pipeline"
            value="Random Forest"
            subtitle="47 Server Features / MediaPipe"
            icon={Cpu}
            badge="FastAPI Interface"
          />
        </section>

        {/* Primary Monitoring Viewport & Telemetry Grid */}
        <section className="dashboard-grid">
          {/* Left Column: Camera Viewport with MediaPipe & FastAPI Streaming */}
          <CameraPanel
            onPredictionUpdate={handlePredictionUpdate}
            onPredictionError={handlePredictionError}
            onCameraStatusChange={handleCameraStatusChange}
          />

          {/* Right Column: Fall Status & Recent Activity */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <FallStatusCard
              livePrediction={livePrediction}
              isLiveActive={cameraActive}
              apiError={apiError}
            />
            <IncidentLog />
          </div>
        </section>

        {/* Historical Alert Audit Table */}
        <section style={{ marginTop: '1.5rem' }}>
          <AlertHistory />
        </section>

        {/* Academic Architecture & Specifications */}
        <SystemInfo />
      </main>

      {/* Footer */}
      <footer className="app-footer" role="contentinfo">
        <div>
          <strong>Elderly Fall Detection &amp; Alert System</strong> &bull; Academic Project (MacBook Air M4 + Windows Team)
        </div>
        <div>
          React 18 &bull; Vite 5 &bull; Prepared for MediaPipe Pose &amp; FastAPI Backend
        </div>
      </footer>
    </div>
  );
}
