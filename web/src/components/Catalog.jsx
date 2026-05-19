import React, { useEffect, useMemo, useState } from 'react';
import { THEME, tempColor, pHColor, saltColor } from '../theme.js';
import { MediaConfBar, OxygenConfArc, IntervalBar, MonoTag, SourceBadge } from './Primitives.jsx';

function ModeStrip({ mode, setMode }) {
  const focused = mode === 'focused';
  return (
    <div style={{ display: 'flex', borderBottom: `1px solid ${THEME.rule}`, background: focused ? '#ede4cd' : THEME.paper, transition: 'background 200ms' }}>
      <button onClick={() => setMode('focused')} style={{
        flex: 1, padding: '14px 28px', textAlign: 'left', border: 'none', cursor: 'pointer',
        background: focused ? '#ede4cd' : 'transparent',
        borderBottom: focused ? `2px solid ${THEME.accent}` : '2px solid transparent',
        font: `400 13px ${THEME.font}`, color: focused ? THEME.ink : THEME.inkFaint,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ font: `500 12px ${THEME.mono}`, color: focused ? THEME.accent : THEME.inkFaint, letterSpacing: '0.05em' }}>1,294</span>
          <span style={{ fontWeight: focused ? 500 : 400 }}>truly never-cultured</span>
          {focused && <span style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkSoft, marginLeft: 'auto' }}>NCBI name starts with "uncultured"</span>}
        </div>
      </button>
      <button onClick={() => setMode('broad')} style={{
        flex: 1, padding: '14px 28px', textAlign: 'left', border: 'none', cursor: 'pointer',
        background: !focused ? '#e8e0c8' : 'transparent',
        borderBottom: !focused ? `2px solid ${THEME.accent}` : '2px solid transparent',
        borderLeft: `1px solid ${THEME.rule}`,
        font: `400 13px ${THEME.font}`, color: !focused ? THEME.ink : THEME.inkFaint,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ font: `500 12px ${THEME.mono}`, color: !focused ? THEME.accent : THEME.inkFaint, letterSpacing: '0.05em' }}>5,000</span>
          <span style={{ fontWeight: !focused ? 500 : 400 }}>all candidates</span>
          {!focused && <span style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkSoft, marginLeft: 'auto' }}>includes 3,706 named-but-absent-from-BacDive</span>}
        </div>
      </button>
    </div>
  );
}

const FILTERS = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'thermo', label: 'Thermophiles', sub: '>55°C', test: (m) => m.T_opt > 55 },
  { id: 'psychro', label: 'Psychrophiles', sub: '<15°C', test: (m) => m.T_opt < 15 },
  { id: 'anaerobe', label: 'Anaerobes', test: (m) => (m.O2 || '').toLowerCase().includes('anaerobe') },
  { id: 'halo', label: 'Halotolerant', sub: '>3% NaCl', test: (m) => m.salt > 3 },
];

function QuickFilters({ active, setActive, count }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '16px 28px 12px', flexWrap: 'wrap' }}>
      {FILTERS.map((f) => (
        <button key={f.id} onClick={() => setActive(f.id)} style={{
          padding: '6px 12px', border: `1px solid ${active === f.id ? THEME.ink : THEME.rule}`,
          background: active === f.id ? THEME.ink : 'transparent',
          color: active === f.id ? THEME.paper : THEME.ink,
          font: `400 12px ${THEME.font}`, cursor: 'pointer', borderRadius: 2,
          display: 'inline-flex', alignItems: 'baseline', gap: 6,
        }}>
          {f.label}
          {f.sub && <span style={{ font: `400 11px ${THEME.mono}`, opacity: 0.7 }}>{f.sub}</span>}
        </button>
      ))}
      <div style={{ marginLeft: 'auto', font: `400 12px ${THEME.mono}`, color: THEME.inkFaint }}>
        showing <span style={{ color: THEME.ink, fontWeight: 500 }}>{count}</span> candidates · sorted by top-medium confidence
      </div>
    </div>
  );
}

function PhenoMicro({ label, value, unit = '', color }) {
  // Synthetic interval: ±5% of scale to show the primitive without per-row data
  // (Catalog has point estimates only; real intervals show in Predict + Detail)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</span>
        <span style={{ font: `500 13px ${THEME.serif}`, color: THEME.ink, fontVariantNumeric: 'tabular-nums' }}>{value}{unit}</span>
      </div>
    </div>
  );
}

function FeaturedCard({ m, onSelect }) {
  return (
    <div onClick={() => onSelect(m)} style={{
      background: THEME.paper, border: `1px solid ${THEME.rule}`, padding: '16px 18px',
      cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: 12, transition: 'border-color 120ms',
    }} onMouseEnter={(e) => (e.currentTarget.style.borderColor = THEME.ink)}
       onMouseLeave={(e) => (e.currentTarget.style.borderColor = THEME.rule)}>
      <div>
        <div style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkFaint, letterSpacing: '0.02em', marginBottom: 4 }}>
          {m.accession} · {m.phylum}
        </div>
        <div style={{ font: `500 14.5px ${THEME.serif}`, color: THEME.ink, lineHeight: 1.25, fontStyle: 'italic' }}>
          {m.name}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 14px', padding: '10px 0', borderTop: `1px solid ${THEME.ruleSoft}`, borderBottom: `1px solid ${THEME.ruleSoft}` }}>
        <PhenoMicro label="T_opt" value={m.T_opt.toFixed(0)} unit="°C" color={tempColor(m.T_opt)} />
        <PhenoMicro label="pH" value={m.pH.toFixed(1)} color={pHColor(m.pH)} />
        <PhenoMicro label="salt" value={m.salt.toFixed(1)} unit="%" color={saltColor(m.salt)} />
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, letterSpacing: '0.05em', textTransform: 'uppercase' }}>O₂</span>
            <SourceBadge source={m.O2_source} compact />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <OxygenConfArc value={m.O2_conf} size={26} />
            <div style={{ font: `500 12px ${THEME.font}`, color: THEME.ink, lineHeight: 1.2 }}>{m.O2}</div>
          </div>
        </div>
      </div>
      <div>
        <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 6 }}>
          Try this medium
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <MonoTag>{m.top_medium_id}</MonoTag>
          <div style={{ flex: 1, font: `400 13px ${THEME.font}`, color: THEME.ink, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.top_medium_name}</div>
        </div>
        <MediaConfBar value={m.top_confidence} />
      </div>
    </div>
  );
}

function TableRow({ m, onSelect, isLast }) {
  return (
    <tr onClick={() => onSelect(m)} style={{
      cursor: 'pointer', borderBottom: isLast ? 'none' : `1px solid ${THEME.ruleSoft}`,
    }} onMouseEnter={(e) => (e.currentTarget.style.background = '#ede5cd')}
       onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}>
      <td style={{ padding: '10px 12px', font: `400 11px ${THEME.mono}`, color: THEME.inkSoft, whiteSpace: 'nowrap' }}>{m.accession}</td>
      <td style={{ padding: '10px 12px', font: `400 13px ${THEME.serif}`, fontStyle: 'italic', color: THEME.ink }}>{m.name}</td>
      <td style={{ padding: '10px 12px', font: `400 11.5px ${THEME.font}`, color: THEME.inkSoft }}>{m.phylum}</td>
      <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>
        <MonoTag>{m.top_medium_id}</MonoTag>
        <span style={{ font: `400 12px ${THEME.font}`, color: THEME.ink, marginLeft: 8 }}>{m.top_medium_name}</span>
      </td>
      <td style={{ padding: '10px 12px' }}><MediaConfBar value={m.top_confidence} compact /></td>
      <td style={{ padding: '10px 12px', font: `500 12px ${THEME.serif}`, color: tempColor(m.T_opt), fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>{m.T_opt.toFixed(0)}°C</td>
      <td style={{ padding: '10px 12px', font: `500 12px ${THEME.serif}`, color: pHColor(m.pH), fontVariantNumeric: 'tabular-nums' }}>{m.pH.toFixed(1)}</td>
      <td style={{ padding: '10px 12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <OxygenConfArc value={m.O2_conf} size={20} />
          <span style={{ font: `400 11.5px ${THEME.font}`, color: THEME.ink }}>{m.O2}</span>
          <SourceBadge source={m.O2_source} compact />
        </div>
      </td>
      <td style={{ padding: '10px 12px', font: `500 12px ${THEME.serif}`, color: saltColor(m.salt), fontVariantNumeric: 'tabular-nums' }}>{m.salt.toFixed(1)}%</td>
      <td style={{ padding: '10px 12px', font: `400 11px ${THEME.mono}`, color: THEME.inkSoft, fontVariantNumeric: 'tabular-nums' }}>{m.completeness.toFixed(0)}</td>
    </tr>
  );
}

export default function Catalog({ data, onSelect }) {
  const [mode, setMode] = useState('focused');
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    let list = data || [];
    if (mode === 'focused') list = list.filter((m) => m.truly_uncultured);
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((m) => m.name.toLowerCase().includes(q) || m.accession.toLowerCase().includes(q));
    }
    const f = FILTERS.find((x) => x.id === filter)?.test || (() => true);
    list = list.filter(f);
    return [...list].sort((a, b) => b.top_confidence - a.top_confidence);
  }, [data, mode, filter, search]);

  const featured = filtered.slice(0, 6);
  const rest = filtered.slice(6, 86);

  return (
    <div style={{ flex: 1, overflow: 'auto', background: THEME.paper }}>
      <ModeStrip mode={mode} setMode={setMode} />
      <QuickFilters active={filter} setActive={setFilter} count={filtered.length} />

      <div style={{ padding: '0 28px 16px', display: 'flex', alignItems: 'center', gap: 12, borderBottom: `1px solid ${THEME.ruleSoft}` }}>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', border: `1px solid ${THEME.rule}`, background: 'rgba(255,255,255,0.4)', maxWidth: 360 }}>
          <span style={{ font: `400 12px ${THEME.mono}`, color: THEME.inkFaint }}>⌕</span>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="search organism name…"
            style={{ flex: 1, border: 'none', background: 'transparent', outline: 'none', font: `400 13px ${THEME.font}`, color: THEME.ink }} />
        </div>
        <button style={{ font: `400 12px ${THEME.mono}`, color: THEME.accent, background: 'transparent', border: `1px solid ${THEME.accent}`, padding: '6px 10px', cursor: 'pointer', borderRadius: 2, marginLeft: 'auto' }}>
          ↓ download CSV
        </button>
      </div>

      {featured.length > 0 && (
        <div style={{ padding: '20px 28px 8px' }}>
          <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>Top {featured.length} picks</span>
            <span style={{ flex: 1, height: 1, background: THEME.rule }} />
            <span style={{ color: THEME.inkFaint, textTransform: 'none', letterSpacing: '0.02em' }}>by media confidence</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
            {featured.map((m) => <FeaturedCard key={m.accession} m={m} onSelect={onSelect} />)}
          </div>
        </div>
      )}

      {rest.length > 0 && (
        <div style={{ padding: '20px 28px 28px' }}>
          <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span>Remaining {filtered.length - 6}</span>
            <span style={{ flex: 1, height: 1, background: THEME.rule }} />
          </div>
          <div style={{ border: `1px solid ${THEME.rule}`, background: THEME.paper, overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: THEME.paperDeep, borderBottom: `1px solid ${THEME.rule}` }}>
                  {['Accession', 'Organism', 'Phylum', 'Try this medium', 'Conf.', 'T', 'pH', 'O₂', 'Salt', 'CheckM'].map((h) => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', font: `500 10px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.05em', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rest.map((m, i) => <TableRow key={m.accession} m={m} onSelect={onSelect} isLast={i === rest.length - 1} />)}
              </tbody>
            </table>
          </div>
          {filtered.length > 86 && (
            <div style={{ textAlign: 'center', padding: '10px 0', font: `400 11px ${THEME.mono}`, color: THEME.inkFaint }}>
              showing first 80 of {filtered.length - 6} remaining · use search and filters to narrow
            </div>
          )}
        </div>
      )}
    </div>
  );
}
