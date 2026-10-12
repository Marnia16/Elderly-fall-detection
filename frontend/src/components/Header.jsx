import React, { useState, useEffect } from 'react';
import { Shield, Activity, Wifi, WifiOff, Clock } from 'lucide-react';
import { API_BASE_URL, checkBackendHealth } from '../config/api';

export default function Header() {
  const [time, setTime] = useState(new Date().toLocaleTimeString());
  const [backendState, setBackendState] = useState({
    checked: false,
    online: false,
    status: 'checking',
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let mounted = true;
    checkBackendHealth().then((res) => {
      if (mounted) {
        setBackendState({
          checked: true,
          online: res.online,
          status: res.status,
        });
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <header className="app-header" role="banner">
      <div className="header-brand">
        <div className="brand-icon-wrapper" aria-hidden="true">
          <Shield size={22} />
        </div>
        <div>
          <h1 className="brand-title">Elderly Fall Detection & Alert System</h1>
          <p className="brand-subtitle">
            AI-Assisted Vision Telemetry &bull; Academic Prototype
          </p>
        </div>
      </div>

      <div className="header-status-group" aria-label="System status overview">
        {/* Live Clock */}
        <div className="badge badge-mock" title="Local System Time">
          <Clock size={13} aria-hidden="true" />
          <span className="mono">{time}</span>
        </div>

        {/* Backend Connectivity Status */}
        <div
          className={`badge ${
            backendState.online ? 'badge-backend-online' : 'badge-backend-offline'
          }`}
          title={`Backend Endpoint: ${API_BASE_URL}`}
        >
          {backendState.online ? (
            <>
              <Wifi size={13} aria-hidden="true" />
              <span className="pulse-dot" aria-hidden="true"></span>
              <span>FastAPI Connected</span>
            </>
          ) : (
            <>
              <WifiOff size={13} aria-hidden="true" />
              <span>FastAPI Standby ({backendState.status})</span>
            </>
          )}
        </div>

        {/* UI Mock Notice Badge */}
        <div className="badge badge-mock" title="Demonstration mode active">
          <Activity size={13} aria-hidden="true" />
          <span>Demo / Mock Data Active</span>
        </div>
      </div>
    </header>
  );
}
