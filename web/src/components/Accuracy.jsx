import React from 'react';
import { THEME, tempColor, pHColor, saltColor, O2_COLOR } from '../theme.js';
import { MediaConfBar, OxygenConfArc, IntervalBar } from './Primitives.jsx';

const TARGETS = [
  { key: 'T', label: 'Temperature optimum', metric: 'MAE', value: '3.28', unit: '°C', color: tempColor(45),
    verdict: "Useful — labs incubate in 5°C steps; you'd usually pick the right shelf.",
    detail: 'Trained on 17,007 BacDive strains. Cross-validation 5-fold GroupKFold by family.' },
  { key: 'pH', label: 'pH optimum', metric: 'MAE', value: '0.52', unit: '', color: pHColor(7),
    verdict: 'Marginal — distinguishes acidic / neutral / alkaline, not finer.',
    detail: 'Trained on 4,652 BacDive strains. Quantile regression for 80% prediction interval.' },
  { key: 'O2', label: 'Oxygen requirement', metric: 'F1', value: '0.28', unit: '', color: O2_COLOR,
    verdict: 'Weak — 9 imbalanced classes, frequent aerobe ↔ aerotolerant confusion.',
    detail: 'Trained on 10,426 BacDive strains. Multinomial logistic on top of XGBoost.' },
  { key: 'salt', label: 'Salt tolerance', metric: 'MAE', value: '2.51', unit: '%', color: saltColor(3),
    verdict: 'Decent — separates freshwater / marine / halotolerant.',
    detail: 'Trained on 4,793 BacDive strains.' },
];

export default function Accuracy() {
  return (
    <div style={{ flex: 1, overflow: 'auto', background: THEME.paper, padding: '24px 28px' }}>
      <div style={{ marginBottom: 24, padding: '14px 16px', background: '#ede4cd', border: `1px solid ${THEME.rule}` }}>
        <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.accent, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: 4 }}>The verdict</div>
        <div style={{ font: `400 14px ${THEME.serif}`, color: THEME.ink, fontStyle: 'italic' }}>
          v1 handcrafted features is the working baseline. Trust temperature and pH; verify oxygen and salt with a tube.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 24 }}>
        {TARGETS.map((t) => (
          <div key={t.key} style={{ border: `1px solid ${THEME.rule}`, padding: '16px 18px', background: THEME.paper }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <span style={{ font: `500 12px ${THEME.font}`, color: THEME.ink }}>{t.label}</span>
              <span style={{ font: `400 11px ${THEME.mono}`, color: THEME.inkFaint }}>5-fold GroupKFold by family</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 8 }}>
              <span style={{ font: `400 10px ${THEME.mono}`, color: THEME.inkFaint, textTransform: 'uppercase' }}>{t.metric}</span>
              <span style={{ font: `500 32px ${THEME.serif}`, color: t.color, fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{t.value}</span>
              <span style={{ font: `400 12px ${THEME.mono}`, color: THEME.inkSoft }}>{t.unit}</span>
            </div>
            <div style={{ font: `500 13px ${THEME.serif}`, color: THEME.ink, fontStyle: 'italic', marginBottom: 6 }}>"{t.verdict}"</div>
            <div style={{ font: `400 12px ${THEME.font}`, color: THEME.inkSoft, lineHeight: 1.5 }}>{t.detail}</div>
          </div>
        ))}
      </div>

      <div style={{ borderTop: `1px solid ${THEME.rule}`, paddingTop: 18 }}>
        <div style={{ font: `500 11px ${THEME.mono}`, color: THEME.inkSoft, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 10 }}>
          <span>How confidence is calculated</span>
          <span style={{ flex: 1, height: 1, background: THEME.rule }} />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <Legend kind="media" />
          <Legend kind="oxygen" />
          <Legend kind="interval" />
        </div>
      </div>

      <div style={{ marginTop: 24, font: `400 12px ${THEME.font}`, color: THEME.inkSoft, lineHeight: 1.6 }}>
        Trained on <span style={{ color: THEME.ink, fontWeight: 500 }}>17,047</span> BacDive strains
        with growth conditions; uncultured catalog is <span style={{ color: THEME.ink, fontWeight: 500 }}>5,000</span> held-out
        GTDB genomes scored against <span style={{ color: THEME.ink, fontWeight: 500 }}>24</span> DSMZ media.
        Features: 353 handcrafted genome statistics — GC, codon usage, tetranucleotide frequencies, amino-acid composition.
        XGBoost classifiers for media; quantile regression XGBoost for prediction intervals.
      </div>
    </div>
  );
}

function Legend({ kind }) {
  if (kind === 'media') {
    return (
      <div style={{ border: `1px solid ${THEME.rule}`, padding: '12px 14px', background: THEME.paper }}>
        <MediaConfBar value={0.72} />
        <div style={{ font: `500 12px ${THEME.font}`, color: THEME.ink, marginTop: 8 }}>Media confidence</div>
        <div style={{ font: `400 11.5px ${THEME.font}`, color: THEME.inkSoft, lineHeight: 1.5 }}>
          Per-medium binary classifier <span style={{ font: `400 11px ${THEME.mono}` }}>predict_proba</span>. Not perfectly calibrated — BacDive only has positive examples.
        </div>
      </div>
    );
  }
  if (kind === 'oxygen') {
    return (
      <div style={{ border: `1px solid ${THEME.rule}`, padding: '12px 14px', background: THEME.paper }}>
        <OxygenConfArc value={0.72} size={36} />
        <div style={{ font: `500 12px ${THEME.font}`, color: THEME.ink, marginTop: 8 }}>Oxygen confidence</div>
        <div style={{ font: `400 11.5px ${THEME.font}`, color: THEME.inkSoft, lineHeight: 1.5 }}>
          Max softmax probability across 9 imbalanced classes. Low values mean the model can't pick between near-neighbour categories.
        </div>
      </div>
    );
  }
  return (
    <div style={{ border: `1px solid ${THEME.rule}`, padding: '12px 14px', background: THEME.paper }}>
      <IntervalBar value={37} lo={32} hi={43} scaleMin={0} scaleMax={80} color={tempColor(37)} unit="°C" height={6} label />
      <div style={{ font: `500 12px ${THEME.font}`, color: THEME.ink, marginTop: 6 }}>Prediction interval</div>
      <div style={{ font: `400 11.5px ${THEME.font}`, color: THEME.inkSoft, lineHeight: 1.5 }}>
        Quantile regression at α=0.1 / 0.9 → 80% PI for T, pH, salt. Wide interval = model uncertain.
      </div>
    </div>
  );
}
