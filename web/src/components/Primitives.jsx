import React from 'react';
import { THEME, O2_COLOR } from '../theme.js';

// Filled bar — media classifier predict_proba
export function MediaConfBar({ value, color = THEME.accent, height = 8, compact = false }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: compact ? 88 : 120 }}>
      <div style={{ flex: 1, height, background: 'rgba(0,0,0,0.06)', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${pct}%`, background: color, borderRadius: 2 }} />
      </div>
      <span style={{ fontFamily: THEME.mono, fontVariantNumeric: 'tabular-nums', fontSize: 12, fontWeight: 500, minWidth: 36, textAlign: 'right' }}>
        {pct}%
      </span>
    </div>
  );
}

// Segmented donut arc — oxygen softmax max prob
export function OxygenConfArc({ value, size = 36, color = O2_COLOR }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  const r = size / 2 - 3;
  const c = 2 * Math.PI * r;
  return (
    <div style={{ position: 'relative', width: size, height: size, flexShrink: 0, display: 'inline-block', verticalAlign: 'middle' }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={THEME.ruleSoft} strokeWidth="3" strokeDasharray="2 2" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="3" strokeDasharray={`${(c * value).toFixed(2)} ${c.toFixed(2)}`} />
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: THEME.mono, fontSize: size > 30 ? 10 : 9, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
        {pct}
      </div>
    </div>
  );
}

// Range bar with point-estimate marker — quantile regression interval
export function IntervalBar({ value, lo, hi, scaleMin, scaleMax, color, unit = '', height = 6, label = false, width = '100%' }) {
  const range = (scaleMax - scaleMin) || 1;
  const loPct = Math.max(0, Math.min(100, ((lo - scaleMin) / range) * 100));
  const hiPct = Math.max(0, Math.min(100, ((hi - scaleMin) / range) * 100));
  const valPct = Math.max(0, Math.min(100, ((value - scaleMin) / range) * 100));
  return (
    <div style={{ width }}>
      <div style={{ height, background: THEME.ruleSoft, borderRadius: 999, position: 'relative' }}>
        <div style={{ position: 'absolute', left: `${loPct}%`, width: `${hiPct - loPct}%`, top: 0, bottom: 0, background: color, opacity: 0.28, borderRadius: 999 }} />
        <div style={{ position: 'absolute', left: `${valPct}%`, top: -2, bottom: -2, width: 2, background: color, transform: 'translateX(-1px)' }} />
      </div>
      {label && (
        <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: THEME.mono, fontSize: 10, color: THEME.inkFaint, marginTop: 3, fontVariantNumeric: 'tabular-nums' }}>
          <span>{lo}{unit}</span>
          <span style={{ color: THEME.ink, fontWeight: 600 }}>{value}{unit}</span>
          <span>{hi}{unit}</span>
        </div>
      )}
    </div>
  );
}

export function MonoTag({ children, color = THEME.accent }) {
  return (
    <span style={{
      fontFamily: THEME.mono, fontSize: 11, fontWeight: 500,
      padding: '1px 6px', border: `1px solid ${color}`, color, borderRadius: 2, whiteSpace: 'nowrap',
    }}>{children}</span>
  );
}

export function SourceBadge({ source = 'tabular', compact = false }) {
  const normalized = String(source || 'tabular').toLowerCase();
  const isLora = normalized === 'lora';
  const color = isLora ? O2_COLOR : THEME.inkFaint;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      width: 'fit-content',
      fontFamily: THEME.mono,
      fontSize: compact ? 9 : 10,
      fontWeight: 500,
      letterSpacing: '0.04em',
      textTransform: 'uppercase',
      padding: compact ? '1px 4px' : '2px 6px',
      border: `1px solid ${color}`,
      color,
      borderRadius: 2,
      whiteSpace: 'nowrap',
      lineHeight: 1.2,
    }}>
      {isLora ? 'LoRA' : 'tabular'}
    </span>
  );
}
