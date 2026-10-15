import React, { useState, useEffect } from 'react';
import {
  BellRing,
  Check,
  Search,
  Filter,
  Download,
  Printer,
  Eye,
  CheckCircle2,
  Clock,
  X,
  RefreshCw,
} from 'lucide-react';
import {
  fetchAlerts,
  acknowledgeAlert,
  resolveAlert,
  getAlertsExportCsvUrl,
} from '../config/api';
import { MOCK_ALERT_HISTORY } from '../mock/mockData';

export default function AlertHistory({ refreshTrigger = 0, onIncidentChanged = () => {} }) {
  const [alerts, setAlerts] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [actionLoadingId, setActionLoadingId] = useState(null);

  const loadAlerts = () => {
    setLoading(true);
    fetchAlerts({
      status: statusFilter,
      severity: severityFilter,
      search: searchQuery || undefined,
      limit: 50,
    })
      .then((data) => {
        if (Array.isArray(data)) {
          setAlerts(data);
          setIsLive(true);
        }
      })
      .catch(() => {
        setIsLive(false);
        // Fallback to mock data with client filtering
        let filtered = [...MOCK_ALERT_HISTORY];
        if (severityFilter !== 'ALL') {
          filtered = filtered.filter(
            (item) => item.severity?.toUpperCase() === severityFilter
          );
        }
        setAlerts(filtered);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    loadAlerts();
  }, [statusFilter, severityFilter, refreshTrigger]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    loadAlerts();
  };

  const handleQuickAcknowledge = async (id) => {
    setActionLoadingId(id);
    try {
      await acknowledgeAlert(id, 'Quick acknowledged by caregiver on dashboard.');
      loadAlerts();
      onIncidentChanged();
    } catch (err) {
      alert(`Acknowledge failed: ${err.message}`);
    } finally {
      setActionLoadingId(null);
    }
  };

  const handleQuickResolve = async (id) => {
    const note = prompt('Enter resolution notes for this incident:', 'Resident evaluated safe and supported.');
    if (note === null) return; // User cancelled
    setActionLoadingId(id);
    try {
      await resolveAlert(id, note);
      loadAlerts();
      onIncidentChanged();
    } catch (err) {
      alert(`Resolve failed: ${err.message}`);
    } finally {
      setActionLoadingId(null);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const formatDateTime = (raw) => {
    if (!raw) return 'N/A';
    try {
      const d = new Date(raw);
      return d.toLocaleString([], {
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return raw;
    }
  };

  return (
    <div className="card-panel" role="region" aria-label="Historical Alert Log">
      <div className="panel-title-bar" style={{ flexWrap: 'wrap', gap: '0.75rem' }}>
        <h2 className="panel-title">
          <BellRing size={19} className="metric-icon" aria-hidden="true" />
          <span>Caregiver Incident Audit History</span>
        </h2>

        {/* Filter and Action Controls */}
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <span
            className="badge"
            style={{
              backgroundColor: isLive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.15)',
              color: isLive ? '#34d399' : '#94a3b8',
              border: isLive ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(148, 163, 184, 0.3)',
            }}
          >
            {isLive ? '[SQLite Real Records]' : '[Mock Records]'}
          </span>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              backgroundColor: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.3rem 0.6rem',
              fontSize: '0.78rem',
            }}
            aria-label="Filter alerts by status"
          >
            <option value="ALL">All Statuses</option>
            <option value="ACTIVE">Active</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="RESOLVED">Resolved</option>
          </select>

          {/* Severity Filter */}
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
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
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>

          {/* Search Box */}
          <form onSubmit={handleSearchSubmit} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
            <input
              type="text"
              placeholder="Search incidents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.3rem 0.5rem',
                fontSize: '0.78rem',
                width: '130px',
              }}
            />
            <button
              type="submit"
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.5rem', fontSize: '0.75rem' }}
              title="Search"
            >
              <Search size={12} />
            </button>
          </form>

          {/* CSV Download & Print Buttons */}
          <a
            href={getAlertsExportCsvUrl(statusFilter)}
            download="incidents_export.csv"
            className="btn btn-secondary"
            style={{
              fontSize: '0.75rem',
              padding: '0.3rem 0.6rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              textDecoration: 'none',
            }}
            title="Download incident report as CSV"
          >
            <Download size={13} />
            Export CSV
          </a>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handlePrint}
            style={{
              fontSize: '0.75rem',
              padding: '0.3rem 0.6rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
            }}
            title="Print printable report view"
          >
            <Printer size={13} />
            Print
          </button>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Incident ID</th>
              <th scope="col">Timestamp</th>
              <th scope="col">Subject &amp; Location</th>
              <th scope="col">Severity</th>
              <th scope="col">Description / Type</th>
              <th scope="col">Confidence / Prob</th>
              <th scope="col">Status</th>
              <th scope="col">Audit Actions</th>
            </tr>
          </thead>
          <tbody>
            {alerts.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  {loading ? 'Loading incidents from database...' : 'No incident records match the selected filters.'}
                </td>
              </tr>
            ) : (
              alerts.map((row) => {
                const id = row.id;
                const timestamp = row.created_at || row.timestamp;
                const severity = (row.severity || 'HIGH').toUpperCase();
                const status = (row.status || 'RESOLVED').toUpperCase();
                const message = row.message || row.eventType || 'Fall event detected';
                const patientId = row.patient_id || 'patient-101';
                const location = row.location || 'Living Room';
                const conf = row.confidence ? `${(row.confidence * 100).toFixed(0)}%` : 'N/A';
                const fallProb = row.fall_probability ? `${(row.fall_probability * 100).toFixed(0)}%` : conf;

                return (
                  <tr key={id}>
                    <td className="mono" style={{ fontWeight: 600, color: 'var(--accent-cyan)' }}>
                      <div>{id}</div>
                      <span
                        className="badge"
                        style={{
                          fontSize: '0.62rem',
                          padding: '0.1rem 0.35rem',
                          marginTop: '2px',
                          display: 'inline-block',
                          backgroundColor: row.is_demo ? 'rgba(234, 179, 8, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                          color: row.is_demo ? '#facc15' : '#34d399',
                          border: row.is_demo ? '1px solid rgba(234, 179, 8, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
                          fontWeight: 600,
                        }}
                      >
                        {row.is_demo ? 'Demo Seed' : 'Live Telemetry'}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                      {formatDateTime(timestamp)}
                    </td>
                    <td style={{ fontSize: '0.8rem' }}>
                      <strong>{patientId}</strong> &bull; {location}
                    </td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          backgroundColor:
                            severity === 'CRITICAL' || severity === 'HIGH'
                              ? 'var(--status-danger-bg)'
                              : severity === 'MEDIUM'
                              ? 'var(--status-warn-bg)'
                              : 'rgba(56, 189, 248, 0.1)',
                          color:
                            severity === 'CRITICAL' || severity === 'HIGH'
                              ? 'var(--status-danger-text)'
                              : severity === 'MEDIUM'
                              ? 'var(--status-warn-text)'
                              : 'var(--accent-cyan)',
                          textTransform: 'uppercase',
                        }}
                      >
                        {severity}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.8rem', maxWidth: '280px' }}>
                      <div style={{ fontWeight: 600, color: '#f8fafc' }}>
                        {row.prediction_type || 'CONFIRMED_FALL'}
                      </div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.74rem' }}>
                        {message}
                      </div>
                    </td>
                    <td style={{ fontSize: '0.78rem' }}>
                      <div>Conf: <strong>{conf}</strong></div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.72rem' }}>Prob: {fallProb}</div>
                    </td>
                    <td>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 4,
                          color:
                            status === 'ACTIVE'
                              ? '#f87171'
                              : status === 'ACKNOWLEDGED'
                              ? '#fb923c'
                              : 'var(--status-safe-text)',
                          fontSize: '0.8rem',
                          fontWeight: 600,
                        }}
                      >
                        {status === 'RESOLVED' && <Check size={13} aria-hidden="true" />}
                        {status}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                        {/* Details Modal Trigger */}
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={() => setSelectedIncident(row)}
                          style={{ padding: '0.2rem 0.45rem', fontSize: '0.72rem' }}
                          title="View telemetry snapshot & details"
                        >
                          <Eye size={12} />
                        </button>

                        {/* Quick Acknowledge */}
                        {status === 'ACTIVE' && (
                          <button
                            type="button"
                            className="btn btn-secondary"
                            disabled={actionLoadingId === id}
                            onClick={() => handleQuickAcknowledge(id)}
                            style={{
                              padding: '0.2rem 0.45rem',
                              fontSize: '0.72rem',
                              backgroundColor: '#ea580c',
                              color: '#fff',
                              borderColor: '#fb923c',
                            }}
                            title="Acknowledge active alarm"
                          >
                            Ack
                          </button>
                        )}

                        {/* Quick Resolve */}
                        {status !== 'RESOLVED' && (
                          <button
                            type="button"
                            className="btn btn-primary"
                            disabled={actionLoadingId === id}
                            onClick={() => handleQuickResolve(id)}
                            style={{
                              padding: '0.2rem 0.45rem',
                              fontSize: '0.72rem',
                              backgroundColor: '#16a34a',
                              borderColor: '#22c55e',
                            }}
                            title="Resolve incident"
                          >
                            Resolve
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Incident Details Modal */}
      {selectedIncident && (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.75)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1100,
            padding: '1rem',
          }}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-secondary, #1e293b)',
              border: '1px solid var(--border-subtle, #334155)',
              borderRadius: 'var(--radius-lg, 12px)',
              padding: '1.5rem',
              maxWidth: '560px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#f8fafc' }}>
                Incident Details &bull; <span style={{ color: 'var(--accent-cyan)' }}>{selectedIncident.id}</span>
              </h3>
              <button
                type="button"
                onClick={() => setSelectedIncident(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ fontSize: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div>
                <strong>Subject:</strong> {selectedIncident.patient_id || 'patient-101'} &bull; <strong>Location:</strong> {selectedIncident.location || 'Living Room'}
              </div>
              <div>
                <strong>Timestamp:</strong> {selectedIncident.created_at || selectedIncident.timestamp}
              </div>
              <div>
                <strong>Status:</strong>{' '}
                <span style={{ fontWeight: 700 }}>{selectedIncident.status}</span>
                {selectedIncident.acknowledged_at && ` (Ack: ${selectedIncident.acknowledged_at})`}
                {selectedIncident.resolved_at && ` (Resolved: ${selectedIncident.resolved_at})`}
              </div>
              <div>
                <strong>Severity:</strong> {selectedIncident.severity} &bull; <strong>Prediction Type:</strong> {selectedIncident.prediction_type}
              </div>
              <div>
                <strong>Confidence:</strong> {selectedIncident.confidence ?? 'N/A'} &bull; <strong>Fall Prob:</strong> {selectedIncident.fall_probability ?? 'N/A'} &bull; <strong>Duration:</strong> {selectedIncident.duration_seconds ?? 0}s
              </div>
              <div>
                <strong>Message:</strong> {selectedIncident.message}
              </div>
              {selectedIncident.notes && (
                <div style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', padding: '0.6rem', borderRadius: '4px' }}>
                  <strong>Caregiver Notes:</strong>
                  <div style={{ marginTop: '0.25rem', color: '#cbd5e1' }}>{selectedIncident.notes}</div>
                </div>
              )}
              {selectedIncident.sensor_snapshot && (
                <div>
                  <strong>Pose &amp; Sensor Telemetry Snapshot:</strong>
                  <pre
                    style={{
                      backgroundColor: '#0f172a',
                      padding: '0.6rem',
                      borderRadius: '4px',
                      fontSize: '0.72rem',
                      overflowX: 'auto',
                      color: '#38bdf8',
                      marginTop: '0.3rem',
                    }}
                  >
                    {JSON.stringify(selectedIncident.sensor_snapshot, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div style={{ marginTop: '1.25rem', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setSelectedIncident(null)}
              >
                Close Details
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
