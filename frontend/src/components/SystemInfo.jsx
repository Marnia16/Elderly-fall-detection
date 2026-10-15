import React from 'react';
import { Info, Terminal, Cpu, GitBranch } from 'lucide-react';
import { API_BASE_URL } from '../config/api';

export default function SystemInfo() {
  return (
    <div
      style={{
        backgroundColor: 'rgba(15, 23, 42, 0.5)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        padding: '1.25rem 1.5rem',
        marginTop: '1.5rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem',
          marginBottom: '0.75rem',
          color: 'var(--accent-cyan)',
        }}
      >
        <Info size={18} aria-hidden="true" />
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Academic Project Architecture &amp; Integration Specs
        </h3>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '1rem',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            Frontend &amp; Vision (Client)
          </div>
          <p>
            React + Vite running on MacBook Air M4. Pre-configured for MediaPipe Pose (33 keypoints)
            and Vercel deployment.
          </p>
        </div>

        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            Backend &amp; Inference API
          </div>
          <p>
            Python + FastAPI developed by teammate (Windows) targeting Render. Endpoint configurable
            via <code className="mono">VITE_API_BASE_URL</code> (Current:{' '}
            <span className="mono" style={{ color: 'var(--accent-cyan)' }}>
              {API_BASE_URL}
            </span>
            ).
          </p>
        </div>

        <div>
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            ML Classifier (Upcoming Phase)
          </div>
          <p>
            Random Forest trained offline on public fall-detection dataset. Zero pre-trained fall
            black boxes or fabricated metric claims.
          </p>
        </div>
      </div>
    </div>
  );
}
