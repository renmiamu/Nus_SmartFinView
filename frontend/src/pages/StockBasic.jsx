// StockBasic.jsx – Black & White Theme with NavBar & Richer Content
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export default function StockBasic() {
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
  const [navVisible, setNavVisible] = useState(false);

  // Animate nav items
  useEffect(() => {
    const timer = setTimeout(() => setNavVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleSearch = () => {
    if (ticker) navigate(`/stock/${ticker}`);
  };

  const popular = ['AAPL', 'TSLA', 'MSFT', 'GOOG'];

  return (
    <div style={{ position: 'relative', minHeight: '100vh', background: '#fff', color: '#000', overflowX: 'hidden' }}>
      {/* Navbar styles */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to   { transform: translateX(0); opacity: 1; }
        }
        .navbar { position: fixed; top:0; width:100%; background:#111; display:flex; justify-content:center; padding:16px 0; z-index:1000; }
        .nav-item { color:#aaa; margin:0 24px; font-size:16px; font-weight:500; position:relative; padding-bottom:6px; cursor:pointer; opacity:0; transition:color .3s; }
        .nav-item:hover, .nav-item.active { color:#fff; }
        .nav-item.active::after { content:''; position:absolute; bottom:0; left:0; width:100%; height:2px; background:#fff; }
        .section-title { font-size:20px; margin:32px 0 16px; }
        .ticker-chip { display:inline-block; margin:4px; padding:6px 12px; border:1px solid #000; border-radius:12px; cursor:pointer; font-size:14px; }
        .ticker-chip:hover { background:#000; color:#fff; }
      `}</style>

      {/* Navigation */}
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

      {/* Main content - centered */}
      <div style={{
        marginTop: '80px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 80px)',
        padding: 20,
        zIndex: 1
      }}>
        <h2>Basic Stock Information</h2>
        <input
          type="text"
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter ticker symbol (e.g. AAPL)"
          style={{ padding:10, border:'1px solid #000', borderRadius:6, width:300, marginBottom:12, fontSize:16, outline:'none' }}
          onKeyDown={e => e.key==='Enter' && handleSearch()}
        />
        <button
          onClick={handleSearch}
          style={{ background:'#000', color:'#fff', border:'none', borderRadius:6, padding:'10px 24px', fontSize:16, cursor:'pointer', marginBottom:24 }}
        >Search</button>

        {/* Popular Tickers directly below search */}
        <div style={{ textAlign:'center', width:'100%' }}>
          <div className="section-title">Popular Tickers</div>
          {popular.map(sym => (
            <span key={sym} className="ticker-chip" onClick={() => navigate(`/stock/${sym}`)}>
              {sym}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
