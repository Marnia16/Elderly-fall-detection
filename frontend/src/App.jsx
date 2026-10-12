import React from 'react';
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
  return (
    <div className="app-container">
      {/* Top Navigation & Live Telemetry Header */}
      <Header />

      {/* Prominent Demonstration & Simulation Disclaimer */}
      <div className="disclaimer-banner" role="status">
        <div className="disclaimer-content">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{MOCK_DISCLAIMER_TEXT}</span>
        </div>
        <span style={{ fontSize: '0.72rem', opacity: 0.85 }}>
          [Demonstration Foundation Mode]
        </span>
      </div>

      {/* Main Dashboard Layout */}
      <main className="main-content">
        {/* Metric Overview Grid */}
        <section className="metrics-grid" aria-label="System Metrics Overview">
          <MetricCard
            title="Camera Input"
            value={MOCK_SYSTEM_METRICS.camera.status}
            subtitle={MOCK_SYSTEM_METRICS.camera.resolution}
            icon={Camera}
            badge="Webcam Ready"
          />

          <MetricCard
            title="Current Fall State"
            value="Safe / Upright"
            subtitle="Posture confidence: 98.4%"
            icon={Activity}
            badge="Mock Telemetry"
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
            subtitle="MediaPipe 33-Keypoint Ready"
            icon={Cpu}
            badge="FastAPI Interface"
          />
        </section>

        {/* Primary Monitoring Viewport & Telemetry Grid */}
        <section className="dashboard-grid">
          {/* Left Column: Camera Viewport Placeholder */}
          <CameraPanel />

          {/* Right Column: Fall Status & Recent Activity */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <FallStatusCard />
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
