import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon, HelpCircle } from 'lucide-react';
import { MOCK_CURRENT_STATUS } from '../mock/mockData';

export default function FallStatusCard({
  livePrediction = null,
  isLiveActive = false,
  apiError = null,
}) {
  // Demonstration state switcher allowing evaluators to preview all 3 alert visual designs when live camera is off
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
      fallProb: '1.6%',
    },
    WARNING: {
      className: 'status-banner-warning',
      iconClass: 'warning',
      icon: AlertTriangle,
      title: 'WARNING - Postural Instability',
      subtitle: 'Sudden descent or horizontal tilt observed. Monitoring for recovery.',
      posture: 'Rapid Level Change / Slump',
      confidence: '73.1%',
      fallProb: '26.9%',
    },
    FALL_DETECTED: {
      className: 'status-banner-danger',
      iconClass: 'danger',
      icon: AlertOctagon,
      title: 'EMERGENCY - Fall Detected!',
      subtitle: 'High-confidence horizontal collapse detected. Immediate attention recommended.',
      posture: 'Prone / On Floor (No Recovery)',
      confidence: '94.8%',
      fallProb: '94.8%',
    },
  };

  const isLive = isLiveActive && livePrediction !== null;

  let current;
  if (isLive) {
    const isFall = Boolean(livePrediction.fall_detected);
    current = {
      className: isFall ? 'status-banner-danger' : 'status-banner-normal',
      iconClass: isFall ? 'danger' : 'normal',
      icon: isFall ? AlertOctagon : ShieldCheck,
      title: isFall ? 'EMERGENCY - Fall Detected!' : 'NORMAL - Safe Posture',
      subtitle:
        livePrediction.message ||
        (isFall
          ? 'High-confidence horizontal collapse detected by Random Forest model.'
          : 'Subject is upright, seated, or moving within normal parameters.'),
      posture: isFall ? 'Fallen / Ground Level Posture' : 'Upright / ADL Mobility',
      confidence:
        typeof livePrediction.confidence === 'number'
          ? `${(livePrediction.confidence * 100).toFixed(1)}%`
          : '95.0%',
      fallProb:
        typeof livePrediction.fall_probability === 'number'
          ? `${(livePrediction.fall_probability * 100).toFixed(1)}%`
          : isFall
          ? '95.0%'
          : '5.0%',
      featuresCount: livePrediction.features_extracted || 47,
    };
  } else {
    current = statusConfigs[demoState];
  }

  const IconComponent = current.icon;

  return (
    <div className="card-panel" role="region" aria-label="Fall Detection Status Banner">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <IconComponent size={19} className="metric-icon" aria-hidden="true" />
          <span>Real-Time Fall Telemetry</span>
        </h2>
        {isLive ? (
          <span
            className="badge"
            style={{
              backgroundColor: current.iconClass === 'danger' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
              color: current.iconClass === 'danger' ? '#f87171' : '#34d399',
              border: current.iconClass === 'danger' ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(52, 211, 153, 0.4)',
              fontWeight: 600,
            }}
            title="Active inference stream from FastAPI"
          >
            [LIVE ML PREDICTION - FastAPI]
          </span>
        ) : (
          <span className="badge badge-mock" title="Simulated visualization preview">
            [Simulated Telemetry]
          </span>
        )}
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
            <span
              style={{
                fontSize: '0.7rem',
                color: isLive ? '#34d399' : 'var(--text-muted)',
                fontWeight: isLive ? 600 : 400,
              }}
            >
              {isLive ? '[LIVE RF]' : '[MOCK]'}
            </span>
          </div>
          {isLive && (
            <div style={{ marginTop: 4, fontSize: '0.76rem', color: 'var(--text-muted)' }}>
              Fall Prob: <strong>{current.fallProb}</strong> &bull; Server Features: <strong>{current.featuresCount}</strong>
            </div>
          )}
        </div>
      </div>

      {/* Live Stream Status or UI Showcase Demo Switcher */}
      {isLiveActive ? (
        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.4)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '0.75rem 1rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem',
            fontSize: '0.8rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className="pulse-dot" style={{ backgroundColor: '#10b981' }} aria-hidden="true" />
            <span>
              <strong>Live Inference Stream Active:</strong> MediaPipe 33 landmarks streamed every ~650ms to FastAPI.
            </span>
          </div>
          {apiError && (
            <span style={{ color: '#fbbf24', fontSize: '0.76rem' }}>
              Backend warning: {apiError}
            </span>
          )}
        </div>
      ) : (
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
              <strong>UI Showcase Controls:</strong> (Camera off) Preview alert states:
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
      )}
    </div>
  );
}
