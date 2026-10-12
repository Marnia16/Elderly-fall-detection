import React, { useState } from 'react';
import { BellRing, Check, Search, Filter } from 'lucide-react';
import { MOCK_ALERT_HISTORY } from '../mock/mockData';

export default function AlertHistory() {
  const [filter, setFilter] = useState('ALL');

  const filteredData = MOCK_ALERT_HISTORY.filter((item) => {
    if (filter === 'ALL') return true;
    return item.severity.toUpperCase() === filter;
  });

  return (
    <div className="card-panel" role="region" aria-label="Historical Alert Log">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <BellRing size={19} className="metric-icon" aria-hidden="true" />
          <span>Historical Alert Audit Log</span>
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="badge badge-mock">[Mock Records]</span>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.3rem 0.6rem',
              fontSize: '0.78rem',
            }}
            aria-label="Filter alerts by severity"
          >
            <option value="ALL">All Severities</option>
            <option value="MEDIUM">Medium Severity</option>
            <option value="LOW">Low Severity</option>
          </select>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Event ID</th>
              <th scope="col">Timestamp</th>
              <th scope="col">Incident Description</th>
              <th scope="col">Severity</th>
              <th scope="col">Dispatch Target</th>
              <th scope="col">Resolution</th>
              <th scope="col">Audit Notes</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map((row) => (
              <tr key={row.id}>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>
                  {row.id}
                </td>
                <td className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {row.timestamp}
                </td>
                <td>{row.eventType}</td>
                <td>
                  <span
                    className="badge"
                    style={{
                      backgroundColor:
                        row.severity === 'high'
                          ? 'var(--status-danger-bg)'
                          : row.severity === 'medium'
                          ? 'var(--status-warn-bg)'
                          : 'rgba(56, 189, 248, 0.1)',
                      color:
                        row.severity === 'high'
                          ? 'var(--status-danger-text)'
                          : row.severity === 'medium'
                          ? 'var(--status-warn-text)'
                          : 'var(--accent-cyan)',
                      textTransform: 'uppercase',
                    }}
                  >
                    {row.severity}
                  </span>
                </td>
                <td>{row.notified}</td>
                <td>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 4,
                      color: 'var(--status-safe-text)',
                      fontSize: '0.8rem',
                    }}
                  >
                    <Check size={13} aria-hidden="true" />
                    {row.status} ({row.responseTime})
                  </span>
                </td>
                <td style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>{row.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
