import React from 'react';

export default function MetricCard({ title, value, subtitle, icon: Icon, badge }) {
  return (
    <div className="metric-card" role="region" aria-label={title}>
      <div className="metric-header">
        <span className="metric-title">{title}</span>
        {Icon && <Icon className="metric-icon" size={18} aria-hidden="true" />}
      </div>
      <div className="metric-value">{value}</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="metric-subtitle">{subtitle}</span>
        {badge && (
          <span
            style={{
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              fontStyle: 'italic',
            }}
          >
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}
