import React from 'react';
import { THEME, tempColor, pHColor, saltColor } from '../theme.js';
import { MediaConfBar, OxygenConfArc, IntervalBar, MonoTag, SourceBadge } from './Primitives.jsx';

export default function DetailDrawer({ microbe, onClose }) {
  if (!microbe) return null;
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(31,29,24,0.4)', zIndex: 50,
      display: 'flex', justifyContent: 'flex-end',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 520, maxWidth: '100%', height: '100%', background: THEME.paper,
        borderLeft: `1px solid ${THEME.ink}`, padding: '20px 24px', overflow: 'auto',
        boxShadow: '-12px 0 24px rgba(0,0,0,0.08)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
          <div>
            <div style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkFaint, marginBottom: 4 }}>
              {microbe.accession} · CheckM {microbe.completeness.toFixed(1)}%
            </div>
            <div style={{ font: `500 17px ${THEME.serif}`, fontStyle: 'italic', color: THEME.ink, lineHeight: 1.3 }}>
              {microbe.name}
            </div>
            <div style={{ font: `400 12px ${THEME.font}`, color: THEME.inkSoft, marginTop: 4 }}>
              {microbe.phylum}
            </div>
          </div>
          <button onClick={onClose} style={{ border: 'none', background: 'transparent', font: `400 18px ${THEME.mono}`, color: THEME.inkSoft, cursor: 'pointer' }}>×</button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 18 }}>
          <PhenoCell label="T_opt" value={microbe.T_opt} unit="°C" color={tempColor(microbe.T_opt)} scaleMin={0} scaleMax={110} />
          <PhenoCell label="pH" value={microbe.pH} color={pHColor(microbe.pH)} scaleMin={2} scaleMax={11} />
          <PhenoCell label="salt" value={microbe.salt} unit="%" color={saltColor(microbe.salt)} scaleMin={0} scaleMax={25} />
          <div style={{ border: `1px solid ${THEME.rule}`, padding: '10px 12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <span style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Oxygen</span>
              <SourceBadge source={microbe.O2_source} compact />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <OxygenConfArc value={microbe.O2_conf} size={36} />
              <div style={{ font: `500 14px ${THEME.font}`, color: THEME.ink }}>{microbe.O2}</div>
            </div>
          </div>
        </div>

        <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
          Top media to try
        </div>
        {[
          { id: microbe.top_medium_id, name: microbe.top_medium_name, conf: microbe.top_confidence },
          { id: microbe.top2_medium_id, name: microbe.top2_medium_name, conf: microbe.top2_confidence },
          { id: microbe.top3_medium_id, name: microbe.top3_medium_name, conf: microbe.top3_confidence },
        ].filter((x) => x.id).map((m, i) => (
          <div key={m.id} style={{
            border: `1px solid ${i === 0 ? THEME.accent : THEME.rule}`,
            padding: '12px 14px',
            background: i === 0 ? '#fdf6e8' : THEME.paper,
            marginBottom: 8, borderRadius: 2,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <MonoTag>{m.id}</MonoTag>
              <span style={{ font: `500 13px ${THEME.font}`, color: THEME.ink, flex: 1 }}>{m.name}</span>
              <MediaConfBar value={m.conf} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PhenoCell({ label, value, unit = '', color, scaleMin, scaleMax }) {
  return (
    <div style={{ border: `1px solid ${THEME.rule}`, padding: '10px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
        <span style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
        <span style={{ font: `500 17px ${THEME.serif}`, color: THEME.ink, fontVariantNumeric: 'tabular-nums' }}>{value}{unit}</span>
      </div>
      <IntervalBar value={value} lo={Math.max(scaleMin, value - (scaleMax - scaleMin) * 0.05)} hi={Math.min(scaleMax, value + (scaleMax - scaleMin) * 0.05)} scaleMin={scaleMin} scaleMax={scaleMax} color={color} height={5} />
    </div>
  );
}
