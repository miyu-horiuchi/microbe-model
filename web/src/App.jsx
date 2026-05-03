import React, { useEffect, useState } from 'react';
import Header, { TabBar } from './components/Header.jsx';
import Catalog from './components/Catalog.jsx';
import PredictBar from './components/PredictBar.jsx';
import TestTab from './components/TestTab.jsx';
import Accuracy from './components/Accuracy.jsx';
import DetailDrawer from './components/DetailDrawer.jsx';

export default function App() {
  const [tab, setTab] = useState('catalog');
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [predictResult, setPredictResult] = useState(null);

  useEffect(() => {
    fetch('/api/catalog')
      .then((r) => r.json())
      .then((d) => setCatalog(d.rows))
      .catch((e) => setError(String(e.message || e)));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f5f1e8', display: 'flex', flexDirection: 'column' }}>
      <Header />
      <TabBar tab={tab} setTab={setTab} />

      {tab === 'catalog' && (
        <>
          <PredictBar result={predictResult} setResult={setPredictResult} />
          {error && <div style={{ padding: '10px 28px', background: '#f5d8c8' }}>{error}</div>}
          {!catalog && !error && (
            <div style={{ padding: '40px 28px', textAlign: 'center', fontFamily: '"IBM Plex Mono", monospace', color: '#94907f' }}>
              loading catalog…
            </div>
          )}
          {catalog && <Catalog data={catalog} onSelect={setSelected} />}
        </>
      )}
      {tab === 'test' && <TestTab />}
      {tab === 'accuracy' && <Accuracy />}

      <DetailDrawer microbe={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
