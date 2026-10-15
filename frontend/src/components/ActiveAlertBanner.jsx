import React, { useState, useEffect } from 'react';
import {
  AlertOctagon,
  Volume2,
  VolumeX,
  CheckCircle,
  Clock,
  MapPin,
  User,
  ShieldAlert,
  X,
} from 'lucide-react';
import { acknowledgeAlert, resolveAlert } from '../config/api';
import { isAudioMuted, setAudioMuted, startAlarm, stopAlarm } from '../utils/audioAlert';

export default function ActiveAlertBanner({
  activeAlerts = [],
  onAlertUpdated = () => {},
  livePrediction = null,
}) {
  const [muted, setMutedState] = useState(isAudioMuted());
  const [resolvingAlertId, setResolvingAlertId] = useState(null);
  const [resolveNotes, setResolveNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [actionError, setActionError] = useState(null);

  // If there are any active alerts in the database, prioritize the most severe/recent
  const primaryAlert = activeAlerts.length > 0 ? activeAlerts[0] : null;

  // Check if live prediction reports active confirmed fall or immobility
  const isLiveFallConfirmed = Boolean(livePrediction?.fall_confirmed || livePrediction?.fall_detected);
  const hasPostFallImmobility = Boolean(livePrediction?.post_fall_immobility);

  const shouldTriggerAlarm = Boolean(primaryAlert || isLiveFallConfirmed);

  // Trigger or stop audio alert based on active fall status and mute setting
  useEffect(() => {
    if (shouldTriggerAlarm && !muted) {
      startAlarm();
    } else {
      stopAlarm();
    }
    return () => {
      stopAlarm();
    };
  }, [shouldTriggerAlarm, muted]);

  const handleToggleMute = () => {
    const nextMuted = !muted;
    setAudioMuted(nextMuted);
    setMutedState(nextMuted);
  };

  const handleAcknowledge = async (alertId) => {
    setIsSubmitting(true);
    setActionError(null);
    try {
      await acknowledgeAlert(alertId, 'Caregiver acknowledged active fall notification.');
      stopAlarm();
      onAlertUpdated();
    } catch (err) {
      setActionError(err.message || 'Failed to acknowledge alert.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOpenResolveModal = (alertId) => {
    setResolvingAlertId(alertId);
    setResolveNotes('Caregiver arrived on scene. Resident verified safe.');
  };

  const handleConfirmResolve = async (e) => {
    e.preventDefault();
    if (!resolvingAlertId) return;
    setIsSubmitting(true);
    setActionError(null);
    try {
      await resolveAlert(resolvingAlertId, resolveNotes);
      setResolvingAlertId(null);
      setResolveNotes('');
      stopAlarm();
      onAlertUpdated();
    } catch (err) {
      setActionError(err.message || 'Failed to resolve alert.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!primaryAlert && !isLiveFallConfirmed) {
    return null;
  }

  const alertId = primaryAlert ? primaryAlert.id : (livePrediction?.active_incident_id || 'PENDING');
  const location = primaryAlert ? primaryAlert.location : 'Living Room (Front Camera)';
  const patientId = primaryAlert ? primaryAlert.patient_id : 'patient-001';
  const severity = hasPostFallImmobility ? 'CRITICAL' : (primaryAlert ? primaryAlert.severity : 'HIGH');
  const fallProb = primaryAlert?.fall_probability ?? livePrediction?.fall_probability ?? 0.95;
  const durationSec = primaryAlert?.duration_seconds ?? 0;

  return (
    <>
      <div
        className="active-fall-banner"
        role="alert"
        aria-live="assertive"
        style={{
          backgroundColor: '#7f1d1d',
          border: '2px solid #ef4444',
          borderRadius: 'var(--radius-md)',
          padding: '1rem 1.25rem',
          marginBottom: '1.25rem',
          boxShadow: '0 0 25px rgba(239, 68, 68, 0.45)',
          animation: 'pulse-border 2s infinite',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '1rem',
          }}
        >
          {/* Left: Critical Headline & Badges */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
            <div
              style={{
                backgroundColor: '#dc2626',
                borderRadius: '50%',
                padding: '0.6rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                animation: 'bounce 1s infinite',
              }}
            >
              <AlertOctagon size={30} color="#ffffff" aria-hidden="true" />
            </div>

            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                <span
                  style={{
                    fontSize: '1.15rem',
                    fontWeight: 800,
                    color: '#ffffff',
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                  }}
                >
                  {hasPostFallImmobility
                    ? 'CRITICAL ALERT: POST-FALL IMMOBILITY DETECTED'
                    : 'ACTIVE FALL EVENT CONFIRMED'}
                </span>
                <span
                  className="badge"
                  style={{
                    backgroundColor: severity === 'CRITICAL' ? '#991b1b' : '#b91c1c',
                    color: '#fecaca',
                    border: '1px solid #f87171',
                    fontWeight: 700,
                  }}
                >
                  {severity}
                </span>
                <span
                  className="badge"
                  style={{
                    backgroundColor: 'rgba(0, 0, 0, 0.4)',
                    color: '#fed7aa',
                    border: '1px solid #fb923c',
                    fontFamily: 'monospace',
                  }}
                >
                  {alertId}
                </span>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  marginTop: '0.35rem',
                  fontSize: '0.82rem',
                  color: '#fee2e2',
                  flexWrap: 'wrap',
                }}
              >
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <User size={14} /> Subject: <strong>{patientId}</strong>
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <MapPin size={14} /> Location: <strong>{location}</strong>
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <ShieldAlert size={14} /> Fall Prob: <strong>{(fallProb * 100).toFixed(0)}%</strong>
                </span>
                {durationSec > 0 && (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                    <Clock size={14} /> Duration: <strong>{durationSec.toFixed(1)}s</strong>
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Right: Action Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleToggleMute}
              title={muted ? 'Unmute Emergency Siren' : 'Mute Emergency Siren'}
              style={{
                backgroundColor: muted ? 'rgba(0,0,0,0.5)' : '#991b1b',
                color: '#ffffff',
                borderColor: '#f87171',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.8rem',
              }}
            >
              {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
              <span>{muted ? 'Alarm Muted' : 'Alarm Active'}</span>
            </button>

            {primaryAlert && primaryAlert.status === 'ACTIVE' && (
              <button
                type="button"
                className="btn btn-secondary"
                disabled={isSubmitting}
                onClick={() => handleAcknowledge(primaryAlert.id)}
                style={{
                  backgroundColor: '#ea580c',
                  color: '#ffffff',
                  borderColor: '#fb923c',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                }}
              >
                Acknowledge Alert
              </button>
            )}

            {primaryAlert && (
              <button
                type="button"
                className="btn btn-primary"
                disabled={isSubmitting}
                onClick={() => handleOpenResolveModal(primaryAlert.id)}
                style={{
                  backgroundColor: '#16a34a',
                  borderColor: '#22c55e',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                }}
              >
                <CheckCircle size={15} />
                Resolve Incident
              </button>
            )}
          </div>
        </div>

        {actionError && (
          <div style={{ marginTop: '0.5rem', color: '#fca5a5', fontSize: '0.78rem' }}>
            Error: {actionError}
          </div>
        )}
      </div>

      {/* Resolve Incident Caregiver Notes Modal */}
      {resolvingAlertId && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="resolve-modal-title"
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1rem',
          }}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-secondary, #1e293b)',
              border: '1px solid var(--border-subtle, #334155)',
              borderRadius: 'var(--radius-lg, 12px)',
              padding: '1.5rem',
              maxWidth: '480px',
              width: '100%',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 id="resolve-modal-title" style={{ margin: 0, fontSize: '1.1rem', color: '#f8fafc' }}>
                Resolve Incident {resolvingAlertId}
              </h3>
              <button
                type="button"
                onClick={() => setResolvingAlertId(null)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
                aria-label="Close dialog"
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleConfirmResolve}>
              <div style={{ marginBottom: '1rem' }}>
                <label
                  htmlFor="caregiver-notes"
                  style={{ display: 'block', fontSize: '0.82rem', marginBottom: '0.4rem', color: '#cbd5e1' }}
                >
                  Caregiver Clinical / Audit Notes:
                </label>
                <textarea
                  id="caregiver-notes"
                  rows={4}
                  value={resolveNotes}
                  onChange={(e) => setResolveNotes(e.target.value)}
                  placeholder="Enter details of response, patient assessment, or recovery action taken..."
                  required
                  style={{
                    width: '100%',
                    backgroundColor: 'var(--bg-primary, #0f172a)',
                    border: '1px solid var(--border-subtle, #334155)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: '#f8fafc',
                    padding: '0.6rem 0.8rem',
                    fontSize: '0.85rem',
                    resize: 'vertical',
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setResolvingAlertId(null)}
                  disabled={isSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isSubmitting}
                  style={{ backgroundColor: '#16a34a', borderColor: '#22c55e' }}
                >
                  {isSubmitting ? 'Resolving...' : 'Confirm & Mark Resolved'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
