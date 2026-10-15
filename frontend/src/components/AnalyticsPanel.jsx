import React, { useState } from 'react';
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Download,
  Calendar,
} from 'lucide-react';
import { getAlertsExportCsvUrl } from '../config/api';

export default function AnalyticsPanel({
  analyticsData = null,
  isLoading = false,
  onRefresh = () => {},
}) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const fallbackData = {
    total_incidents: 3,
    active_incidents: 0,
    acknowledged_incidents: 0,
    resolved_incidents: 3,
    average_confidence: 0.88,
    average_fall_probability: 0.81,
    daily_trend: [
      { date: '2026-10-09', label: 'Oct 09', count: 0 },
      { date: '2026-10-10', label: 'Oct 10', count: 0 },
      { date: '2026-10-11', label: 'Oct 11', count: 0 },
      { date: '2026-10-12', label: 'Oct 12', count: 0 },
      { date: '2026-10-13', label: 'Oct 13', count: 1 },
      { date: '2026-10-14', label: 'Oct 14', count: 1 },
      { date: '2026-10-15', label: 'Oct 15', count: 1 },
    ],
  };

  const data = analyticsData || fallbackData;
  const isRealData = Boolean(analyticsData);

  const dailyTrend = data.daily_trend || [];
  const maxCount = Math.max(...dailyTrend.map((d) => d.count), 3); // At least 3 for visual headroom

  // SVG Chart Dimensions
  const chartWidth = 580;
  const chartHeight = 150;
  const paddingX = 45;
  const paddingY = 25;
  const plotWidth = chartWidth - paddingX * 2;
  const plotHeight = chartHeight - paddingY * 2;

  // Compute coordinate points
  const points = dailyTrend.map((item, index) => {
    const x = paddingX + (index / Math.max(dailyTrend.length - 1, 1)) * plotWidth;
    const y = chartHeight - paddingY - (item.count / maxCount) * plotHeight;
    return { ...item, x, y };
  });

  const polylineStr = points.map((p) => `${p.x},${p.y}`).join(' ');
  const areaPathStr = points.length > 0
    ? `M ${points[0].x},${chartHeight - paddingY} ` +
      points.map((p) => `L ${p.x},${p.y}`).join(' ') +
      ` L ${points[points.length - 1].x},${chartHeight - paddingY} Z`
    : '';

  return (
    <div className="card-panel" role="region" aria-label="Incident Analytics & 7-Day Trend">
      <div className="panel-title-bar">
        <h2 className="panel-title">
          <BarChart3 size={19} className="metric-icon" aria-hidden="true" />
          <span>Incident Telemetry &amp; 7-Day Trend Analytics</span>
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span
            className="badge"
            style={{
              backgroundColor: isRealData ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.15)',
              color: isRealData ? '#34d399' : '#94a3b8',
              border: isRealData ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(148, 163, 184, 0.3)',
            }}
          >
            {isRealData ? '[SQLite Live Analytics]' : '[Seeded Baseline]'}
          </span>
          <a
            href={getAlertsExportCsvUrl()}
            download="incidents_export.csv"
            className="btn btn-secondary"
            style={{
              fontSize: '0.75rem',
              padding: '0.25rem 0.6rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              textDecoration: 'none',
            }}
            title="Download full incident history as CSV"
          >
            <Download size={13} />
            CSV Report
          </a>
        </div>
      </div>

      {/* Metric Counters Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '0.75rem',
          marginBottom: '1.25rem',
        }}
      >
        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            Total Incidents
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc', marginTop: '0.2rem' }}>
            {data.total_incidents}
          </div>
        </div>

        <div
          style={{
            backgroundColor: data.active_incidents > 0 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(15, 23, 42, 0.5)',
            border: data.active_incidents > 0 ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              fontSize: '0.72rem',
              color: data.active_incidents > 0 ? '#fca5a5' : 'var(--text-secondary)',
              textTransform: 'uppercase',
            }}
          >
            Active Incidents
          </div>
          <div
            style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: data.active_incidents > 0 ? '#ef4444' : '#f8fafc',
              marginTop: '0.2rem',
            }}
          >
            {data.active_incidents}
          </div>
        </div>

        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            Acknowledged
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#fb923c', marginTop: '0.2rem' }}>
            {data.acknowledged_incidents}
          </div>
        </div>

        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            Resolved
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#34d399', marginTop: '0.2rem' }}>
            {data.resolved_incidents}
          </div>
        </div>

        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
            Avg Fall Prob
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#38bdf8', marginTop: '0.2rem' }}>
            {(data.average_fall_probability * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* SVG 7-Day Trend Chart */}
      <div
        style={{
          backgroundColor: 'rgba(15, 23, 42, 0.4)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          padding: '0.75rem 1rem',
          position: 'relative',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem',
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
          }}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <Calendar size={13} /> 7-Day Event Frequency (Real Database Timestamp Aggregation)
          </span>
          {hoveredPoint && (
            <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>
              {hoveredPoint.label}: {hoveredPoint.count} incident{hoveredPoint.count === 1 ? '' : 's'}
            </span>
          )}
        </div>

        <div style={{ width: '100%', overflowX: 'auto' }}>
          <svg
            viewBox={`0 0 ${chartWidth} ${chartHeight}`}
            style={{ width: '100%', height: 'auto', display: 'block', minWidth: '420px' }}
            aria-label="7-Day Incident Trend Line Chart"
          >
            <defs>
              <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.35" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Horizontal Grid lines */}
            {[0, 0.5, 1].map((ratio, i) => {
              const y = chartHeight - paddingY - ratio * plotHeight;
              const value = Math.round(ratio * maxCount);
              return (
                <g key={i}>
                  <line
                    x1={paddingX}
                    y1={y}
                    x2={chartWidth - paddingX}
                    y2={y}
                    stroke="rgba(255, 255, 255, 0.08)"
                    strokeDasharray="3,3"
                  />
                  <text
                    x={paddingX - 8}
                    y={y + 4}
                    fill="#64748b"
                    fontSize="10"
                    textAnchor="end"
                    fontFamily="monospace"
                  >
                    {value}
                  </text>
                </g>
              );
            })}

            {/* Shaded Area */}
            {areaPathStr && <path d={areaPathStr} fill="url(#areaGradient)" />}

            {/* Trend Line */}
            {polylineStr && (
              <polyline
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={polylineStr}
              />
            )}

            {/* Data Circles and Day Labels */}
            {points.map((point, i) => (
              <g
                key={i}
                onMouseEnter={() => setHoveredPoint(point)}
                onMouseLeave={() => setHoveredPoint(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Transparent hover target */}
                <circle cx={point.x} cy={point.y} r="14" fill="transparent" />

                {/* Visible Data Point */}
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={hoveredPoint?.date === point.date ? 6 : 4}
                  fill={point.count > 0 ? '#ef4444' : '#38bdf8'}
                  stroke="#0f172a"
                  strokeWidth="2"
                />

                {/* Count Badge on points with count > 0 */}
                {point.count > 0 && (
                  <text
                    x={point.x}
                    y={point.y - 9}
                    fill="#f87171"
                    fontSize="10"
                    fontWeight="700"
                    textAnchor="middle"
                  >
                    {point.count}
                  </text>
                )}

                {/* Day Label */}
                <text
                  x={point.x}
                  y={chartHeight - 6}
                  fill="#94a3b8"
                  fontSize="10"
                  textAnchor="middle"
                >
                  {point.label}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
    </div>
  );
}
