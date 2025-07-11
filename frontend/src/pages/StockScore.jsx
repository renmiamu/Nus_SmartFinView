// StockScore.jsx – Black & White Theme with NavBar
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, useLocation } from 'react-router-dom';

export default function StockScore() {
  const navigate = useNavigate();
  const location = useLocation();
  const menuItems = [
    { label: 'Basic Info',      path: '/stock' },
    { label: 'Portfolio Viz',   path: '/portfolio' },
    { label: 'AI Score',        path: '/score' },
    { label: 'Sentiment',       path: '/emotion' },
    { label: 'Recommendations', path: '/recommendation' }
  ];

  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [navVisible, setNavVisible] = useState(false);

  // Nav animation
  useEffect(() => {
    const timer = setTimeout(() => setNavVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleScore = async () => {
    if (!ticker) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.get('http://localhost:8000/stock/score', { params: { ticker } });
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.response?.data?.detail || 'Query failed' });
    }
    setLoading(false);
  };

  return (
    <div style={{ background: '#fff', minHeight: '100vh', color: '#000', overflow: 'hidden' }}>
      {/* Navbar */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to   { transform: translateX(0); opacity: 1; }
        }
        .navbar { position:fixed; top:0; width:100%; background:#000; display:flex; justify-content:center; padding:16px 0; z-index:1000; }
        .nav-item { color:#888; margin:0 24px; font-size:16px; font-weight:500; position:relative; padding-bottom:6px; cursor:pointer; opacity:0; transition:color .3s; }
        .nav-item:hover, .nav-item.active { color:#fff; }
        .nav-item.active::after { content:''; position:absolute; bottom:0; left:0; width:100%; height:2px; background:#fff; }
      `}</style>
      <nav className="navbar">
        {menuItems.map((item, idx) => (
          <div
            key={item.path}
            onClick={() => navigate(item.path)}
            className={`nav-item${location.pathname===item.path?' active':''}`}
            style={{ animation: navVisible?`slideInRight .6s ease-out forwards ${idx*0.15}s`:'none' }}
          >{item.label}</div>
        ))}
      </nav>

      {/* Main score section */}
      <div style={{
        marginTop: '50px',
        display:'flex',
        flexDirection:'column',
        alignItems:'center',
        justifyContent:'center',
        minHeight:'calc(100vh - 80px)',
        padding:20,
        zIndex:1
      }}>
        <h2 style={{ marginBottom:24 }}>AI Stock Score</h2>
        <input
          type="text"
          value={ticker}
          onChange={e=>setTicker(e.target.value.toUpperCase())}
          placeholder="Enter ticker (e.g. AAPL)"
          style={{
            padding:10,
            border:'1px solid #000',
            borderRadius:6,
            width:300,
            marginBottom:20,
            fontSize:16,
            outline:'none'
          }}
          onKeyDown={e=>e.key==='Enter'&&handleScore()}
        />
        <button
          onClick={handleScore}
          disabled={loading}
          style={{
            background:'#000',
            color:'#fff',
            border:'none',
            borderRadius:6,
            padding:'10px 32px',
            fontSize:16,
            cursor:'pointer'
          }}
        >{loading?'Loading...':'Get Score'}</button>

        {result && (
          <div style={{ marginTop:24, textAlign:'center', fontSize:18, color: result.error?'#800':'#000' }}>
            {result.error
              ? result.error
              : <>
                  <div>Ticker: {result.ticker}</div>
                  <div>Score: <strong>{result.score}</strong></div>
                </>
            }
          </div>
        )}

        <div style={{ marginTop:24, color:'#666', fontSize:14 }}>
          Examples: AAPL, TSLA, MSFT, GOOG
        </div>
      </div>
    </div>
  );
}
