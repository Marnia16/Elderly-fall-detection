import React, { useState, useEffect } from 'react';
import { History, AlertCircle, CheckCircle2, Info, RefreshCw } from 'lucide-react';
import { fetchAlerts } from '../config/api';
import { MOCK_RECENT_INCIDENTS } from '../mock/mockData';

export default function IncidentLog({ refreshTrigger = 0 }) {
  const [incidents, setIncidents] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    fetchAlerts({ limit: 6 })
      .then((data) => {
        if (mounted && Array.isArray(data) && data.length > 0) {
          setIncidents(data);
          setIsLive(true);
        } else if (mounted) {
          setIncidents(MOCK_RECENT_INCIDENTS);
          setIsLive(false);
        }
      })
      .catch(() => {
        if (mounted) {
          setIncidents(MOCK_RECENT_INCIDENTS);
          setIsLive(false);
        }
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [refreshTrigger]);

  const getIconForSeverity = (severity) => {
    const sev = String(severity || '').toUpperCase();
    if (sev === 'CRITICAL' || sev === 'HIGH' || sev === 'EMERGENCY' || sev === 'DANGER') {
      return <AlertCircle size={15} color="var(--status-danger-text)" aria-hidden="true" />;
    }
    if (sev === 'MEDIUM' || sev === 'WARNING') {
      return <AlertCircle size={15} color="var(--status-warn-text)" aria-hidden="true" />;
    }
    if (sev === 'RESOLVED') {
      return <CheckCircle2 size={15} color="var(--status-safe-text)" aria-hidden="true" />;
    }
    return <Info size={15} color="var(--accent-cyan)" aria-hidden="true" />;
  };

  const formatTimestamp = (rawTs) => {
    if (!rawTs) return 'Just now';
    try {
      const d = new Date(rawTs);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return rawTs;
    }
  };

  return (
    <div className="card-panel" role="region" aria-label="Recent Incidents Log">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <History size={19} className="metric-icon" aria-hidden="true" />
          <span>Recent Activity &amp; Incidents</span>
        </h2>
        <span
          className={`badge ${isLive ? 'badge-backend-online' : 'badge-mock'}`}
          title={isLive ? 'Real persistent incidents from SQLite' : 'Demonstration log history'}
        >
          {isLive ? '[SQLite Live Log]' : '[Mock Log]'}
        </span>
      </div>

      <div className="incident-list">
        {incidents.map((item) => {
          const id = item.id;
          const title = item.prediction_type || item.type || 'Fall Event';
          const severity = item.severity || 'HIGH';
          const status = item.status || 'ACTIVE';
          const duration = item.duration_seconds ?? item.durationSeconds ?? 0;
          const conf = item.confidence
            ? typeof item.confidence === 'number'
              ? `${(item.confidence * 100).toFixed(0)}%`
              : item.confidence
            : '95%';
          const timeStr = formatTimestamp(item.created_at || item.timestamp);

          return (
            <div key={id} className="incident-item">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div>{getIconForSeverity(severity)}</div>
                <div>
                  <div className="incident-title">
                    <span>{title}</span>
                    <span
                      style={{
                        fontSize: '0.68rem',
                        padding: '0.15rem 0.4rem',
                        borderRadius: '4px',
                        backgroundColor: 'rgba(255, 255, 255, 0.08)',
                        color: 'var(--text-secondary)',
                        fontFamily: 'monospace',
                      }}
                    >
                      {id}
                    </span>
                    <span
                      className="badge"
                      style={{
                        fontSize: '0.6rem',
                        padding: '0.08rem 0.3rem',
                        backgroundColor: item.is_demo ? 'rgba(234, 179, 8, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                        color: item.is_demo ? '#facc15' : '#34d399',
                        border: item.is_demo ? '1px solid rgba(234, 179, 8, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
                      }}
                    >
                      {item.is_demo ? 'Demo' : 'Live'}
                    </span>
                  </div>
                  <div className="incident-meta">
                    <span>{timeStr}</span> &bull; <span>Duration: {Number(duration).toFixed(1)}s</span> &bull;{' '}
                    <span
                      style={{
                        color:
                          status === 'ACTIVE'
                            ? '#f87171'
                            : status === 'ACKNOWLEDGED'
                            ? '#fb923c'
                            : '#34d399',
                        fontWeight: 600,
                      }}
                    >
                      {status}
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color:
                      severity.toUpperCase() === 'CRITICAL' || severity.toUpperCase() === 'HIGH'
                        ? 'var(--status-danger-text)'
                        : 'var(--text-secondary)',
                  }}
                >
                  Conf: {conf}
                </span>
                <div style={{ fontSize: '0.68rem', color: isLive ? '#34d399' : 'var(--text-muted)' }}>
                  {isLive ? '[PERSISTENT]' : '[SIMULATED]'}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
