import React from 'react';
import { THEME } from '../theme.js';

export default function Header() {
  return (
    <div style={{ borderBottom: `1px solid ${THEME.rule}`, padding: '18px 28px 16px', background: THEME.paper }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 4 }}>
        <div style={{ position: 'relative', width: 18, height: 18, borderRadius: '50%', background: THEME.accent, top: 3, display: 'inline-block' }}>
          <div style={{ position: 'absolute', inset: 4, borderRadius: '50%', background: THEME.paper }} />
          <div style={{ position: 'absolute', inset: 7, borderRadius: '50%', background: THEME.accent }} />
        </div>
        <h1 style={{ font: `500 22px/1 ${THEME.serif}`, color: THEME.ink, letterSpacing: '-0.01em', margin: 0 }}>
          microbe-model
        </h1>
        <span style={{ font: `400 11px/1 ${THEME.mono}`, color: THEME.inkFaint, letterSpacing: '0.04em' }}>
          v2.0.0 · trained 2026-03-14
        </span>
      </div>
      <p style={{ font: `400 13.5px/1.5 ${THEME.font}`, color: THEME.inkSoft, margin: '8px 0 0', maxWidth: 640 }}>
        Predicted growth conditions for microbes that have never been cultured. Pick one. Try the medium.
        Five thousand candidates from GTDB scored against twenty-four DSMZ media.
      </p>
    </div>
  );
}

export function TabBar({ tab, setTab }) {
  const tabs = [
    { id: 'catalog', label: 'Catalog' },
    { id: 'test', label: 'Test on a known genome' },
    { id: 'accuracy', label: 'Model accuracy' },
  ];
  return (
    <div style={{ display: 'flex', borderBottom: `1px solid ${THEME.rule}`, background: THEME.paper, padding: '0 28px' }}>
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => setTab(t.id)}
          style={{
            padding: '12px 18px',
            border: 'none',
            background: 'transparent',
            font: `${tab === t.id ? 500 : 400} 13px ${THEME.font}`,
            color: tab === t.id ? THEME.ink : THEME.inkFaint,
            borderBottom: tab === t.id ? `2px solid ${THEME.ink}` : '2px solid transparent',
            cursor: 'pointer',
            marginBottom: -1,
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
