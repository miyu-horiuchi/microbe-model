export const THEME = {
  paper: '#f5f1e8',
  paperDeep: '#ece6d6',
  ink: '#1f1d18',
  inkSoft: '#5a554a',
  inkFaint: '#94907f',
  rule: '#d6cdb6',
  ruleSoft: '#e6dfca',
  accent: '#a8521a',
  accentSoft: '#d68f5a',
  accentTint: '#fdf6e8',
  pos: '#3f6b3a',
  warn: '#a8521a',
  cool: '#3b5d6f',
  warm: '#a04020',
  focusedPaper: '#f5f1e8',
  broadPaper: '#efeadf',
  focusedStrip: '#ede4cd',
  broadStrip: '#e8e0c8',
  font: '"IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
  mono: '"IBM Plex Mono", "JetBrains Mono", ui-monospace, monospace',
  serif: '"IBM Plex Serif", Georgia, serif',
};

export const O2_COLOR = '#3a7d6e';

export const tempColor = (t) => {
  if (t < 15) return '#3b82a6';
  if (t < 30) return '#5b8b9c';
  if (t < 45) return '#7d8470';
  if (t < 60) return '#b06a3b';
  return '#a04020';
};

export const pHColor = (p) => {
  if (p < 6) return '#a04020';
  if (p < 7.5) return '#7d8470';
  return '#3b82a6';
};

export const saltColor = (s) => {
  if (s < 1) return '#7d8470';
  if (s < 5) return '#b89048';
  return '#8a5e1f';
};
