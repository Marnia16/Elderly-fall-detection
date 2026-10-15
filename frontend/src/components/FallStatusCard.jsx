import React, { useState } from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  Activity,
  UserX,
} from 'lucide-react';
import { MOCK_CURRENT_STATUS } from '../mock/mockData';

export default function FallStatusCard({
  livePrediction = null,
  isLiveActive = false,
  apiError = null,
}) {
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
    const isFallConfirmed = Boolean(livePrediction.fall_confirmed || livePrediction.fall_detected);
    const isNearFall = Boolean(livePrediction.near_fall) && !isFallConfirmed;
    const hasImmobility = Boolean(livePrediction.post_fall_immobility);

    if (hasImmobility) {
      current = {
        className: 'status-banner-danger',
        iconClass: 'danger',
        icon: UserX,
        title: 'CRITICAL - Post-Fall Immobility!',
        subtitle: 'Subject has remained motionless on floor for >5 seconds after collapse.',
        posture: 'Stationary / Immobility Detected',
        confidence: '99.0%',
        fallProb: '99.0%',
        featuresCount: livePrediction.features_extracted || 47,
        trackerState: livePrediction.tracker_state || 'FALL_ACTIVE',
        isNearFall: false,
        hasImmobility: true,
      };
    } else if (isFallConfirmed) {
      current = {
        className: 'status-banner-danger',
        iconClass: 'danger',
        icon: AlertOctagon,
        title: 'EMERGENCY - Fall Confirmed!',
        subtitle:
          livePrediction.message ||
          'High-confidence horizontal collapse confirmed by Random Forest temporal engine.',
        posture: 'Fallen / Ground Level Posture',
        confidence:
          typeof livePrediction.confidence === 'number'
            ? `${(livePrediction.confidence * 100).toFixed(1)}%`
            : '95.0%',
        fallProb:
          typeof livePrediction.fall_probability === 'number'
            ? `${(livePrediction.fall_probability * 100).toFixed(1)}%`
            : '95.0%',
        featuresCount: livePrediction.features_extracted || 47,
        trackerState: livePrediction.tracker_state || 'FALL_ACTIVE',
        isNearFall: false,
        hasImmobility: false,
      };
    } else if (isNearFall) {
      current = {
        className: 'status-banner-warning',
        iconClass: 'warning',
        icon: AlertTriangle,
        title: 'WARNING - Near-Fall Tilt Detected',
        subtitle: 'Substantial torso deviation (40°-65°) detected without full collapse.',
        posture: 'Postural Tilt / Wobble',
        confidence:
          typeof livePrediction.confidence === 'number'
            ? `${(livePrediction.confidence * 100).toFixed(1)}%`
            : '78.0%',
        fallProb:
          typeof livePrediction.fall_probability === 'number'
            ? `${(livePrediction.fall_probability * 100).toFixed(1)}%`
            : '45.0%',
        featuresCount: livePrediction.features_extracted || 47,
        trackerState: livePrediction.tracker_state || 'MONITORING',
        isNearFall: true,
        hasImmobility: false,
      };
    } else {
      current = {
        className: 'status-banner-normal',
        iconClass: 'normal',
        icon: ShieldCheck,
        title: 'NORMAL - Safe Posture',
        subtitle:
          livePrediction.message || 'Subject is upright, seated, or moving within normal parameters.',
        posture: 'Upright / ADL Mobility',
        confidence:
          typeof livePrediction.confidence === 'number'
            ? `${(livePrediction.confidence * 100).toFixed(1)}%`
            : '95.0%',
        fallProb:
          typeof livePrediction.fall_probability === 'number'
            ? `${(livePrediction.fall_probability * 100).toFixed(1)}%`
            : '5.0%',
        featuresCount: livePrediction.features_extracted || 47,
        trackerState: livePrediction.tracker_state || 'MONITORING',
        isNearFall: false,
        hasImmobility: false,
      };
    }
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
          <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            {current.trackerState && (
              <span
                className="badge"
                style={{
                  backgroundColor:
                    current.trackerState === 'FALL_ACTIVE'
                      ? 'rgba(239, 68, 68, 0.25)'
                      : current.trackerState === 'CONFIRMING'
                      ? 'rgba(249, 115, 22, 0.25)'
                      : 'rgba(56, 189, 248, 0.2)',
                  color:
                    current.trackerState === 'FALL_ACTIVE'
                      ? '#f87171'
                      : current.trackerState === 'CONFIRMING'
                      ? '#fb923c'
                      : '#38bdf8',
                  fontSize: '0.72rem',
                }}
              >
                State: {current.trackerState}
              </span>
            )}
            <span
              className="badge"
              style={{
                backgroundColor:
                  current.iconClass === 'danger'
                    ? 'rgba(239, 68, 68, 0.2)'
                    : current.iconClass === 'warning'
                    ? 'rgba(245, 158, 11, 0.2)'
                    : 'rgba(16, 185, 129, 0.2)',
                color:
                  current.iconClass === 'danger'
                    ? '#f87171'
                    : current.iconClass === 'warning'
                    ? '#fbbf24'
                    : '#34d399',
                border:
                  current.iconClass === 'danger'
                    ? '1px solid rgba(239, 68, 68, 0.4)'
                    : current.iconClass === 'warning'
                    ? '1px solid rgba(245, 158, 11, 0.4)'
                    : '1px solid rgba(52, 211, 153, 0.4)',
                fontWeight: 600,
              }}
              title="Active inference stream from FastAPI"
            >
              [LIVE ML PREDICTION]
            </span>
          </div>
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

      {/* Live Stream Telemetry Info or UI Showcase Switcher */}
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
              <strong>Temporal Confirmation Engine:</strong> 2 consecutive frames in &le;2.5s window
              suppresses transient anomalies &amp; false alarms.
            </span>
          </div>
          {apiError && (
            <span style={{ color: '#fbbf24', fontSize: '0.76rem' }}>
              Backend note: {apiError}
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
