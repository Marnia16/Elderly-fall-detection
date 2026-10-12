import React from 'react';
import { History, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { MOCK_RECENT_INCIDENTS } from '../mock/mockData';

export default function IncidentLog() {
  const getIconForSeverity = (severity) => {
    switch (severity.toLowerCase()) {
      case 'emergency':
      case 'danger':
        return <AlertCircle size={15} color="var(--status-danger-text)" aria-hidden="true" />;
      case 'warning':
        return <AlertCircle size={15} color="var(--status-warn-text)" aria-hidden="true" />;
      case 'resolved':
        return <CheckCircle2 size={15} color="var(--status-safe-text)" aria-hidden="true" />;
      default:
        return <Info size={15} color="var(--accent-cyan)" aria-hidden="true" />;
    }
  };

  return (
    <div className="card-panel" role="region" aria-label="Recent Incidents Log">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <History size={19} className="metric-icon" aria-hidden="true" />
          <span>Recent Activity &amp; Incidents</span>
        </h2>
        <span className="badge badge-mock" title="Demonstration log history">
          [Mock Log]
        </span>
      </div>

      <div className="incident-list">
        {MOCK_RECENT_INCIDENTS.map((item) => (
          <div key={item.id} className="incident-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div>{getIconForSeverity(item.severity)}</div>
              <div>
                <div className="incident-title">
                  <span>{item.type}</span>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      padding: '0.15rem 0.4rem',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(255, 255, 255, 0.08)',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    {item.id}
                  </span>
                </div>
                <div className="incident-meta">
                  <span>{item.timestamp}</span> &bull; <span>Duration: {item.durationSeconds}s</span> &bull;{' '}
                  <span style={{ color: 'var(--text-secondary)' }}>Status: {item.status}</span>
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
              <span
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: item.severity === 'Warning' ? 'var(--status-warn-text)' : 'var(--text-secondary)',
                }}
              >
                Conf: {item.confidence}
              </span>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>[SIMULATED]</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
