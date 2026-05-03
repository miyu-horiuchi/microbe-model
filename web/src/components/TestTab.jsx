import React, { useState } from 'react';
import { THEME, tempColor, pHColor, saltColor } from '../theme.js';
import { OxygenConfArc, IntervalBar, MediaConfBar, MonoTag } from './Primitives.jsx';

const PRESETS = [
  { id: 'ecoli', name: 'Escherichia coli K-12 MG1655', accession: 'GCF_000005845.2',
    published: { T_opt: 37, pH: 7.0, salt: 1, O2: 'facultative anaerobe', medium: 'LB (Luria-Bertani)' } },
  { id: 'bsub', name: 'Bacillus subtilis 168', accession: 'GCF_000009045.1',
    published: { T_opt: 30, pH: 7.0, salt: 2, O2: 'facultative anaerobe', medium: 'LB or Nutrient Broth' } },
  { id: 'tt',   name: 'Thermus thermophilus HB8', accession: 'GCF_000091545.1',
    published: { T_opt: 70, pH: 7.5, salt: 0.5, O2: 'aerobe', medium: 'DSMZ 74 Castenholz TYE' } },
];

export default function TestTab() {
  const [selected, setSelected] = useState(PRESETS[0]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function predict(preset) {
    setSelected(preset);
    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: preset.accession, top_k: 5 }),
      });
      if (!r.ok) throw new Error(await r.text());
      setResult(await r.json());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ flex: 1, overflow: 'auto', background: THEME.paper, padding: '24px 28px' }}>
      <div style={{ font: `400 12px ${THEME.mono}`, color: THEME.inkSoft, marginBottom: 14 }}>
        Sanity-check the model on a microbe with published growth conditions.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 22 }}>
        {PRESETS.map((p) => (
          <button key={p.id} onClick={() => predict(p)} style={{
            textAlign: 'left', padding: '12px 14px',
            background: selected.id === p.id ? '#ede4cd' : THEME.paper,
            border: `1px solid ${selected.id === p.id ? THEME.ink : THEME.rule}`,
            cursor: 'pointer', borderRadius: 2,
          }}>
            <div style={{ font: `500 13.5px ${THEME.serif}`, fontStyle: 'italic', color: THEME.ink, marginBottom: 4 }}>{p.name}</div>
            <div style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkFaint }}>{p.accession}</div>
            <div style={{ display: 'flex', gap: 10, marginTop: 8, font: `400 11px ${THEME.mono}`, color: THEME.inkSoft }}>
              <span>{p.published.T_opt}°C</span>
              <span>pH {p.published.pH}</span>
              <span>{p.published.O2}</span>
            </div>
          </button>
        ))}
      </div>

      {busy && <div style={{ fontFamily: THEME.mono, color: THEME.inkSoft, fontSize: 12 }}>Predicting…</div>}
      {error && <div style={{ padding: '8px 12px', background: '#f5d8c8', border: `1px solid ${THEME.warn}`, fontFamily: THEME.font, fontSize: 12, color: THEME.ink }}>{error}</div>}

      {result && <Comparison preset={selected} result={result} />}
    </div>
  );
}

function CompareCard({ label, pred, lo, hi, pub, unit, color, scaleMin, scaleMax }) {
  const inInterval = pub >= lo && pub <= hi;
  const pubPct = ((pub - scaleMin) / (scaleMax - scaleMin)) * 100;
  return (
    <div style={{ border: `1px solid ${THEME.rule}`, padding: '14px 16px', background: THEME.paper }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
        <span style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.05em', textTransform: 'uppercase' }}>{label}</span>
        <span style={{ font: `500 11px ${THEME.mono}`, color: inInterval ? THEME.pos : THEME.warn }}>
          {inInterval ? '✓ in 80% PI' : '△ outside PI'}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'center' }}>
        <div>
          <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>predicted</div>
          <div style={{ font: `500 22px ${THEME.serif}`, color, fontVariantNumeric: 'tabular-nums' }}>{pred.toFixed(1)}{unit}</div>
          <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, marginTop: 2 }}>{lo.toFixed(1)}{unit} – {hi.toFixed(1)}{unit}</div>
        </div>
        <div>
          <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>published</div>
          <div style={{ font: `500 22px ${THEME.serif}`, color: THEME.ink, fontVariantNumeric: 'tabular-nums' }}>{pub}{unit}</div>
          <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, marginTop: 2 }}>literature</div>
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <IntervalBar value={pred} lo={lo} hi={hi} scaleMin={scaleMin} scaleMax={scaleMax} color={color} height={6} />
        <div style={{ position: 'relative', height: 0, marginTop: -4 }}>
          <div style={{ position: 'absolute', left: `${pubPct}%`, transform: 'translateX(-50%)', top: 8, fontFamily: THEME.mono, fontSize: 9, fontWeight: 500, color: THEME.ink }}>↑ pub</div>
        </div>
      </div>
    </div>
  );
}

function Comparison({ preset, result }) {
  const p = result.phenotypes;
  const pub = preset.published;
  const T = p.optimal_temperature_c;
  const pH = p.optimal_ph;
  const salt = p.salt_tolerance_pct;
  const O2 = p.oxygen_requirement;
  const top = result.media?.[0];

  return (
    <div>
      <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 12 }}>
        Predicted vs published — {preset.name}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {T && <CompareCard label="Optimum temperature" pred={T.prediction} lo={T.low_80} hi={T.high_80} pub={pub.T_opt} unit="°C" color={tempColor(T.prediction)} scaleMin={0} scaleMax={110} />}
        {pH && <CompareCard label="Optimum pH" pred={pH.prediction} lo={pH.low_80} hi={pH.high_80} pub={pub.pH} unit="" color={pHColor(pH.prediction)} scaleMin={2} scaleMax={11} />}
        {salt && <CompareCard label="Salt tolerance" pred={salt.prediction} lo={salt.low_80} hi={salt.high_80} pub={pub.salt} unit="%" color={saltColor(salt.prediction)} scaleMin={0} scaleMax={25} />}
        {O2 && (
          <div style={{ border: `1px solid ${THEME.rule}`, padding: '14px 16px', background: THEME.paper }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
              <span style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Oxygen requirement</span>
              <span style={{ font: `500 11px ${THEME.mono}`, color: O2.prediction === pub.O2 ? THEME.pos : THEME.warn }}>
                {O2.prediction === pub.O2 ? '✓ match' : '△ mismatch'}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'center' }}>
              <div>
                <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>predicted</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <OxygenConfArc value={O2.confidence} size={32} />
                  <div style={{ font: `500 14px ${THEME.font}`, color: THEME.ink }}>{O2.prediction}</div>
                </div>
              </div>
              <div>
                <div style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>published</div>
                <div style={{ font: `500 16px ${THEME.font}`, color: THEME.ink }}>{pub.O2}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{ marginTop: 22, font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
        Top media to try
      </div>
      {result.media?.slice(0, 5).map((m, i) => (
        <div key={m.medium_id} style={{
          border: `1px solid ${i === 0 ? THEME.accent : THEME.rule}`,
          padding: '12px 14px', background: i === 0 ? '#fdf6e8' : THEME.paper, marginBottom: 8, borderRadius: 2,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <MonoTag>{m.medium_id}</MonoTag>
            <span style={{ font: `500 13px ${THEME.font}`, color: THEME.ink, flex: 1 }}>{m.name}</span>
            <MediaConfBar value={m.confidence} />
          </div>
          {m.recipe && (
            <div style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkSoft, lineHeight: 1.4, marginTop: 6 }}>
              {m.recipe}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
