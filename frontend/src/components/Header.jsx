import React, { useState, useEffect } from 'react';
import {
  Shield,
  Activity,
  Wifi,
  WifiOff,
  Clock,
  Database,
  Cpu,
  Mail,
  Volume2,
  VolumeX,
  Camera,
  CameraOff,
} from 'lucide-react';
import { API_BASE_URL, checkBackendHealth } from '../config/api';
import { isAudioMuted, setAudioMuted } from '../utils/audioAlert';

export default function Header({
  cameraActive = false,
  isLiveStreaming = false,
}) {
  const [time, setTime] = useState(new Date().toLocaleTimeString());
  const [muted, setMuted] = useState(isAudioMuted());
  const [backendState, setBackendState] = useState({
    checked: false,
    online: false,
    status: 'checking',
    modelLoaded: false,
    database: null,
    notifications: null,
    latencyMs: null,
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const refreshHealth = () => {
    checkBackendHealth().then((res) => {
      setBackendState({
        checked: true,
        online: res.online,
        status: res.status,
        modelLoaded: res.modelLoaded || false,
        database: res.database || null,
        notifications: res.notifications || null,
        latencyMs: res.latencyMs,
      });
    });
  };

  useEffect(() => {
    refreshHealth();
    // Periodically re-check health every 15 seconds
    const interval = setInterval(refreshHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleMute = () => {
    const next = !muted;
    setAudioMuted(next);
    setMuted(next);
  };

  const isDbConnected = Boolean(backendState.database?.connected);
  const totalIncidents = backendState.database?.total_incidents ?? 0;
  const notifMode = backendState.notifications?.mode || 'demo';

  return (
    <header className="app-header" role="banner">
      <div className="header-brand">
        <div className="brand-icon-wrapper" aria-hidden="true">
          <Shield size={22} />
        </div>
        <div>
          <h1 className="brand-title">Elderly Fall Detection &amp; Alert System</h1>
          <p className="brand-subtitle">
            AI-Assisted Vision Telemetry &bull; Real-Time Pose Estimation &amp; Random Forest Classification
          </p>
        </div>
      </div>

      <div className="header-status-group" aria-label="System status overview">
        {/* Live Clock */}
        <div className="badge badge-mock" title="System Local Clock">
          <Clock size={13} aria-hidden="true" />
          <span className="mono">{time}</span>
        </div>

        {/* Global Siren Audio Mute/Unmute Toggle */}
        <button
          type="button"
          onClick={handleToggleMute}
          className="badge"
          style={{
            cursor: 'pointer',
            backgroundColor: muted ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
            color: muted ? '#f87171' : '#34d399',
            border: muted ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.35rem',
          }}
          title={muted ? 'Unmute Emergency Fall Alarm' : 'Mute Emergency Fall Alarm'}
        >
          {muted ? <VolumeX size={13} /> : <Volume2 size={13} />}
          <span>{muted ? 'Siren Muted' : 'Siren Armed'}</span>
        </button>

        {/* Camera Hardware Status (Derived strictly from browser media stream, never inferred from backend) */}
        <div
          className={`badge ${cameraActive ? 'badge-backend-online' : 'badge-backend-offline'}`}
          title={cameraActive ? 'Browser Webcam Active (1280x720)' : 'Webcam Offline / Inactive'}
        >
          {cameraActive ? (
            <>
              <Camera size={13} aria-hidden="true" />
              <span className="pulse-dot" aria-hidden="true" />
              <span>Webcam Active</span>
            </>
          ) : (
            <>
              <CameraOff size={13} aria-hidden="true" />
              <span>Webcam Off</span>
            </>
          )}
        </div>

        {/* Live Inference Streaming Status */}
        <div
          className="badge"
          style={{
            backgroundColor: isLiveStreaming ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.15)',
            color: isLiveStreaming ? '#34d399' : '#94a3b8',
            border: isLiveStreaming ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(148, 163, 184, 0.3)',
          }}
          title={isLiveStreaming ? 'Streaming 33 MediaPipe landmarks to FastAPI Random Forest' : 'Inference Standby'}
        >
          <Activity size={13} aria-hidden="true" />
          {isLiveStreaming && <span className="pulse-dot" style={{ backgroundColor: '#10b981' }} aria-hidden="true" />}
          <span>{isLiveStreaming ? 'Live Inference' : 'Inference Standby'}</span>
        </div>

        {/* FastAPI Connectivity Status */}
        <div
          className={`badge ${backendState.online ? 'badge-backend-online' : 'badge-backend-offline'}`}
          title={`Backend Endpoint: ${API_BASE_URL} (${backendState.latencyMs ?? 0}ms)`}
        >
          {backendState.online ? (
            <>
              <Wifi size={13} aria-hidden="true" />
              <span className="pulse-dot" aria-hidden="true" />
              <span>FastAPI Online</span>
            </>
          ) : (
            <>
              <WifiOff size={13} aria-hidden="true" />
              <span>FastAPI Standby ({backendState.status})</span>
            </>
          )}
        </div>

        {/* Model Readiness Badge */}
        <div
          className="badge"
          style={{
            backgroundColor: backendState.modelLoaded ? 'rgba(56, 189, 248, 0.15)' : 'rgba(148, 163, 184, 0.15)',
            color: backendState.modelLoaded ? '#38bdf8' : '#94a3b8',
            border: backendState.modelLoaded ? '1px solid rgba(56, 189, 248, 0.3)' : '1px solid rgba(148, 163, 184, 0.3)',
          }}
          title="Trained Random Forest Classifier (47 Pose Features)"
        >
          <Cpu size={13} aria-hidden="true" />
          <span>{backendState.modelLoaded ? 'RF Model Ready' : 'Model Standby'}</span>
        </div>

        {/* SQLite Database Status */}
        <div
          className="badge"
          style={{
            backgroundColor: isDbConnected ? 'rgba(168, 85, 247, 0.15)' : 'rgba(148, 163, 184, 0.15)',
            color: isDbConnected ? '#c084fc' : '#94a3b8',
            border: isDbConnected ? '1px solid rgba(168, 85, 247, 0.3)' : '1px solid rgba(148, 163, 184, 0.3)',
          }}
          title="Persistent SQLite Incidents Database (WAL mode)"
        >
          <Database size={13} aria-hidden="true" />
          <span>{isDbConnected ? `SQLite DB (${totalIncidents})` : 'SQLite Standby'}</span>
        </div>

        {/* Notification Service Status */}
        <div
          className="badge"
          style={{
            backgroundColor: 'rgba(234, 179, 8, 0.15)',
            color: '#facc15',
            border: '1px solid rgba(234, 179, 8, 0.3)',
          }}
          title={`Notification Mode: ${notifMode.toUpperCase()}`}
        >
          <Mail size={13} aria-hidden="true" />
          <span>Notifs: {notifMode === 'smtp' ? 'SMTP Armed' : 'Demo Mode'}</span>
        </div>
      </div>
    </header>
  );
}
