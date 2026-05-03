import React, { useState, useRef } from 'react';
import { THEME } from '../theme.js';
import { MediaConfBar, OxygenConfArc, IntervalBar, MonoTag } from './Primitives.jsx';
import { tempColor, pHColor, saltColor } from '../theme.js';

const QUICK_TRY = [
  { label: 'Thermus thermophilus', value: 'Thermus thermophilus' },
  { label: 'GCF_000005845.2', value: 'GCF_000005845.2' },
  { label: 'Pyrococcus sp. VENT-07', value: 'Pyrococcus' },
];

export default function PredictBar({ result, setResult }) {
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [hits, setHits] = useState(null);
  const fileRef = useRef(null);

  function isAccession(s) {
    const u = s.trim().toUpperCase();
    return u.startsWith('GCA_') || u.startsWith('GCF_');
  }

  async function runPredict(target) {
    setBusy(true);
    setError(null);
    setHits(null);
    try {
      const r = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target, top_k: 8 }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || `HTTP ${r.status}`);
      }
      const data = await r.json();
      setResult(data);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function onSubmit() {
    const q = query.trim();
    if (!q) return;
    if (isAccession(q)) {
      await runPredict(q);
      return;
    }
    // Treat as organism name → NCBI search → user picks
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(`/api/ncbi-search?q=${encodeURIComponent(q)}`);
      const data = await r.json();
      if (!data.hits?.length) {
        setError(`No NCBI Assembly hits for "${q}".`);
      } else if (data.hits.length === 1) {
        await runPredict(data.hits[0].accession);
      } else {
        setHits(data.hits);
      }
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file) {
    if (!file) return;
    const text = await file.text();
    await runPredict(text);
  }

  return (
    <div style={{ background: THEME.accentTint, borderTop: `1px solid ${THEME.rule}`, borderBottom: `1px solid ${THEME.rule}` }}>
      <div style={{ padding: '16px 28px 8px' }}>
        <div style={{
          fontFamily: THEME.mono, fontSize: 10.5, color: THEME.accent,
          letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 500,
          marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{ fontSize: 10 }}>▶</span> Predict a medium
          <span style={{ fontFamily: THEME.font, fontWeight: 400, fontSize: 11.5, color: THEME.inkSoft, letterSpacing: 0, textTransform: 'none', marginLeft: 4 }}>
            paste a name, NCBI accession, FASTA, or upload a genome
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', border: `1px solid ${THEME.rule}`, background: 'rgba(255,255,255,0.6)', borderRadius: 2 }}>
            <span style={{ fontFamily: THEME.mono, fontSize: 12, color: THEME.inkFaint }}>⌕</span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onSubmit()}
              placeholder='e.g. "Thermus thermophilus"   GCF_000008125.1   >my_mag'
              style={{ flex: 1, border: 'none', background: 'transparent', outline: 'none', font: `400 13px ${THEME.font}`, color: THEME.ink, height: 22 }}
            />
          </div>
          <input ref={fileRef} type="file" accept=".fa,.fna,.fasta,.gz" style={{ display: 'none' }} onChange={(e) => onFile(e.target.files?.[0])} />
          <button onClick={() => fileRef.current?.click()} style={{
            padding: '6px 12px', border: `1px solid ${THEME.rule}`, background: 'transparent',
            font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, cursor: 'pointer', borderRadius: 2,
            letterSpacing: '0.05em', textTransform: 'uppercase',
          }}>↑ FASTA</button>
          <button onClick={onSubmit} disabled={busy || !query.trim()} style={{
            padding: '6px 16px',
            border: `1px solid ${THEME.accent}`,
            background: busy || !query.trim() ? 'transparent' : THEME.accent,
            color: busy || !query.trim() ? THEME.accent : THEME.paper,
            font: `500 11px ${THEME.mono}`, cursor: busy || !query.trim() ? 'not-allowed' : 'pointer', borderRadius: 2,
            letterSpacing: '0.05em', textTransform: 'uppercase',
            opacity: busy || !query.trim() ? 0.6 : 1,
          }}>{busy ? 'predicting…' : 'Predict →'}</button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
          <span style={{ fontFamily: THEME.mono, fontSize: 11, color: THEME.inkFaint }}>try:</span>
          {QUICK_TRY.map((q) => (
            <button key={q.label} onClick={() => { setQuery(q.value); runPredict(q.value); }}
              style={{
                padding: '3px 9px', border: `1px solid ${THEME.rule}`, background: THEME.paper,
                font: `400 11px ${THEME.mono}`, color: THEME.ink, cursor: 'pointer', borderRadius: 2,
              }}>{q.label}</button>
          ))}
        </div>

        {error && (
          <div style={{ marginTop: 8, padding: '8px 12px', background: '#f5d8c8', border: `1px solid ${THEME.warn}`, fontFamily: THEME.font, fontSize: 12, color: THEME.ink }}>
            {error}
          </div>
        )}

        {hits && (
          <div style={{ marginTop: 10, padding: '10px 12px', border: `1px solid ${THEME.rule}`, background: THEME.paper }}>
            <div style={{ fontFamily: THEME.mono, fontSize: 11, color: THEME.inkSoft, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
              {hits.length} NCBI matches — pick one
            </div>
            {hits.map((h) => (
              <button key={h.accession} onClick={() => { setHits(null); runPredict(h.accession); }}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  padding: '6px 8px', border: `1px solid transparent`, background: 'transparent',
                  font: `400 12px ${THEME.font}`, color: THEME.ink, cursor: 'pointer', marginBottom: 2,
                }}>
                <span style={{ fontFamily: THEME.mono, fontSize: 11, color: THEME.accent }}>{h.accession}</span>
                {' '}— <span style={{ fontStyle: 'italic' }}>{h.organism}</span>
                {' · '}<span style={{ fontFamily: THEME.mono, fontSize: 11, color: THEME.inkFaint }}>{h.level}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {result && <PredictResultBanner result={result} onClear={() => setResult(null)} />}
    </div>
  );
}

function PredictResultBanner({ result, onClear }) {
  const p = result.phenotypes || {};
  const top = result.media?.[0];
  const T = p.optimal_temperature_c || {};
  const pH = p.optimal_ph || {};
  const O2 = p.oxygen_requirement || {};
  const salt = p.salt_tolerance_pct || {};

  return (
    <div style={{ padding: '14px 28px', borderTop: `1px solid ${THEME.accent}`, background: '#fef3df' }}>
      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 360px', minWidth: 280 }}>
          <div style={{ fontFamily: THEME.mono, fontSize: 10.5, color: THEME.accent, letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 500 }}>
            Prediction · {result.accession} · {result.n_contigs} contigs · {result.n_cds} CDS · GC {(result.gc * 100).toFixed(1)}%
          </div>
          <div style={{ font: `500 15px ${THEME.serif}`, color: THEME.ink, marginTop: 4 }}>
            Try <span style={{ color: THEME.accent }}>{top?.name}</span>
            {top && <> &nbsp;<MonoTag>{top.medium_id}</MonoTag></>}
          </div>
          {top?.recipe && (
            <div style={{ fontFamily: THEME.mono, fontSize: 11, color: THEME.inkSoft, marginTop: 4, lineHeight: 1.4 }}>
              {top.recipe}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          {T.prediction !== undefined && (
            <PhenoStat label="T_opt" value={`${T.prediction.toFixed(1)}°C`} sub={`${T.low_80?.toFixed(1)}–${T.high_80?.toFixed(1)}°C`} color={tempColor(T.prediction)} />
          )}
          {pH.prediction !== undefined && (
            <PhenoStat label="pH" value={pH.prediction.toFixed(2)} sub={`${pH.low_80?.toFixed(2)}–${pH.high_80?.toFixed(2)}`} color={pHColor(pH.prediction)} />
          )}
          {O2.prediction !== undefined && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <OxygenConfArc value={O2.confidence || 0} size={32} />
              <div>
                <div style={{ fontFamily: THEME.mono, fontSize: 9, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em' }}>O₂</div>
                <div style={{ fontFamily: THEME.font, fontSize: 12, color: THEME.ink }}>{O2.prediction}</div>
              </div>
            </div>
          )}
          {salt.prediction !== undefined && (
            <PhenoStat label="salt" value={`${salt.prediction.toFixed(1)}%`} sub={`${salt.low_80?.toFixed(1)}–${salt.high_80?.toFixed(1)}%`} color={saltColor(salt.prediction)} />
          )}
        </div>
        <button onClick={onClear} style={{
          padding: '4px 10px', border: `1px solid ${THEME.rule}`, background: 'transparent',
          font: `400 10.5px ${THEME.mono}`, color: THEME.inkSoft, cursor: 'pointer',
        }}>clear</button>
      </div>
    </div>
  );
}

function PhenoStat({ label, value, sub, color }) {
  return (
    <div style={{ minWidth: 80 }}>
      <div style={{ fontFamily: THEME.mono, fontSize: 9, color: THEME.inkFaint, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
      <div style={{ fontFamily: THEME.serif, fontSize: 17, fontWeight: 500, color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      <div style={{ fontFamily: THEME.mono, fontSize: 10, color: THEME.inkFaint, fontVariantNumeric: 'tabular-nums' }}>{sub}</div>
    </div>
  );
}
