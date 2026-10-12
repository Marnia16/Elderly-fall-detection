import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon, HelpCircle } from 'lucide-react';
import { MOCK_CURRENT_STATUS } from '../mock/mockData';

export default function FallStatusCard() {
  // Demonstration state switcher allowing evaluators to preview all 3 alert visual designs
  const [demoState, setDemoState] = useState('NORMAL');

  const statusConfigs = {
    NORMAL: {
      className: 'status-banner-normal',
      iconClass: 'normal',
      icon: ShieldCheck,
      title: 'NORMAL - Safe Posture',
      subtitle: 'Subject is upright, seated, or moving within normal velocity thresholds.',
      posture: 'Normal Upright Mobility',
      confidence: '98.4%',
    },
    WARNING: {
      className: 'status-banner-warning',
      iconClass: 'warning',
      icon: AlertTriangle,
      title: 'WARNING - Postural Instability',
      subtitle: 'Sudden descent or horizontal tilt observed. Monitoring for recovery.',
      posture: 'Rapid Level Change / Slump',
      confidence: '73.1%',
    },
    FALL_DETECTED: {
      className: 'status-banner-danger',
      iconClass: 'danger',
      icon: AlertOctagon,
      title: 'EMERGENCY - Fall Detected!',
      subtitle: 'High-confidence horizontal collapse detected. Immediate attention recommended.',
      posture: 'Prone / On Floor (No Recovery)',
      confidence: '94.8%',
    },
  };

  const current = statusConfigs[demoState];
  const IconComponent = current.icon;

  return (
    <div className="card-panel" role="region" aria-label="Fall Detection Status Banner">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <IconComponent size={19} className="metric-icon" aria-hidden="true" />
          <span>Real-Time Fall Telemetry</span>
        </h2>
        <span className="badge badge-mock" title="Simulated visualization preview">
          [Simulated Telemetry]
        </span>
      </div>

      <div className={`status-banner ${current.className}`}>
        <div className="status-indicator-left">
          <div className={`status-icon-box ${current.iconClass}`} aria-hidden="true">
            <IconComponent size={26} />
          </div>
          <div>
            <div className="status-headline">{current.title}</div>
            <div className="status-sub">{current.subtitle}</div>
          </div>
        </div>
        <div className="status-meta-right">
          <div>
            Estimated Posture: <strong>{current.posture}</strong>
          </div>
          <div style={{ marginTop: 4 }}>
            Confidence: <strong>{current.confidence}</strong>{' '}
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>[MOCK]</span>
          </div>
        </div>
      </div>

      {/* Demo State Switcher for academic presentation */}
      <div
        style={{
          backgroundColor: 'rgba(15, 23, 42, 0.4)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          padding: '0.85rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
          <HelpCircle size={15} color="var(--accent-cyan)" aria-hidden="true" />
          <span>
            <strong>UI Showcase Controls:</strong> Preview dashboard alert behaviors:
          </span>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            className={`btn btn-secondary ${demoState === 'NORMAL' ? 'btn-active' : ''}`}
            onClick={() => setDemoState('NORMAL')}
            style={{
              borderColor: demoState === 'NORMAL' ? 'var(--status-safe-border)' : undefined,
              color: demoState === 'NORMAL' ? 'var(--status-safe-text)' : undefined,
            }}
          >
            Safe State
          </button>
          <button
            type="button"
            className={`btn btn-secondary ${demoState === 'WARNING' ? 'btn-active' : ''}`}
            onClick={() => setDemoState('WARNING')}
            style={{
              borderColor: demoState === 'WARNING' ? 'var(--status-warn-border)' : undefined,
              color: demoState === 'WARNING' ? 'var(--status-warn-text)' : undefined,
            }}
          >
            Warning State
          </button>
          <button
            type="button"
            className={`btn btn-secondary ${demoState === 'FALL_DETECTED' ? 'btn-active' : ''}`}
            onClick={() => setDemoState('FALL_DETECTED')}
            style={{
              borderColor: demoState === 'FALL_DETECTED' ? 'var(--status-danger-border)' : undefined,
              color: demoState === 'FALL_DETECTED' ? 'var(--status-danger-text)' : undefined,
            }}
          >
            Fall Alert State
          </button>
        </div>
      </div>
    </div>
  );
}
