// StockRecommendation.jsx – Black & White Theme with NavBar
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, useLocation } from 'react-router-dom';

export default function StockRecommendation() {
  const navigate = useNavigate();
  const location = useLocation();
  const menuItems = [
    { label: 'Basic Info', path: '/stock' },
    { label: 'Portfolio Viz', path: '/portfolio' },
    { label: 'AI Score', path: '/score' },
    { label: 'Sentiment', path: '/emotion' },
    { label: 'Recommendations', path: '/recommendation' }
  ];

  const [form, setForm] = useState({
    risk_tolerance: 'medium',
    industry_preference: [],
    investment_amount: '',
    lookback_period: 365
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [navVisible, setNavVisible] = useState(false);

  const industries = ['Technology', 'Healthcare', 'Finance', 'Energy'];

  useEffect(() => {
    const timer = setTimeout(() => setNavVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmit = async () => {
    if (!form.investment_amount) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post('http://localhost:8000/stock/recommendation', form);
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.response?.data?.detail || 'Query failed' });
    }
    setLoading(false);
  };

  const toggleIndustry = (industry) => {
    setForm((prev) => {
      const has = prev.industry_preference.includes(industry);
      return {
        ...prev,
        industry_preference: has
          ? prev.industry_preference.filter(i => i !== industry)
          : [...prev.industry_preference, industry]
      };
    });
  };

  return (
    <div style={{ minHeight: '100vh', background: '#fff', color: '#000', overflow: 'hidden' }}>
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .navbar { position: fixed; top: 0; width: 100%; background: #000; display: flex; justify-content: center; padding: 12px 0; z-index: 1000; }
        .nav-item { color: #777; margin: 0 24px; font-size: 14px; font-weight: 500; opacity: 0; position: relative; padding-bottom: 4px; cursor: pointer; transition: color .3s; }
        .nav-item:hover, .nav-item.active { color: #fff; }
        .nav-item.active::after { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background: #fff; }
        .card { background: #f9f9f9; padding: 24px; border-radius: 8px; max-width: 720px; margin: 100px auto 40px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .btn { background: #000; color: #fff; border: none; border-radius: 6px; padding: 10px 24px; font-size: 16px; cursor: pointer; }
      `}</style>
      <nav className="navbar">
        {menuItems.map((item, idx) => (
          <div
            key={item.path}
            onClick={() => navigate(item.path)}
            className={`nav-item${location.pathname === item.path ? ' active' : ''}`}
            style={{ animation: navVisible ? `slideInRight 0.6s ease-out forwards ${idx * 0.15}s` : 'none' }}
          >{item.label}</div>
        ))}
      </nav>

      <div className="card">
        <h2 style={{ textAlign: 'center', marginBottom: 24 }}>Portfolio Recommendation</h2>

        <div style={{ marginBottom: 16 }}>
          <label>Investment Amount (USD):</label><br />
          <input
            type="number"
            value={form.investment_amount}
            onChange={e => setForm({ ...form, investment_amount: parseFloat(e.target.value) || '' })}
            style={{ padding: 8, borderRadius: 6, border: '1px solid #000', width: '60%' }}
          />
        </div>

        <div style={{ marginBottom: 16 }}>
          <label>Risk Tolerance:</label><br />
          <select
            value={form.risk_tolerance}
            onChange={e => setForm({ ...form, risk_tolerance: e.target.value })}
            style={{ padding: 8, borderRadius: 6, border: '1px solid #000' }}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        <div style={{ marginBottom: 16 }}>
          <label>Industry Preferences:</label><br />
          {industries.map(ind => (
            <label key={ind} style={{ marginRight: 16 }}>
              <input
                type="checkbox"
                checked={form.industry_preference.includes(ind)}
                onChange={() => toggleIndustry(ind)}
              /> {ind}
            </label>
          ))}
        </div>

        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Loading...' : 'Generate Recommendations'}
        </button>

        {result && (
          <div style={{ marginTop: 32 }}>
            {result.error ? (
              <div style={{ color: '#800' }}>{result.error}</div>
            ) : (
              <div>
                <h3 style={{ marginBottom: 12 }}>Recommended Assets & Weights:</h3>
                <ul>
                  {result.recommended_assets.map(({ ticker, weight }, idx) => (
                    <li key={idx}>{ticker} - {weight}%</li>
                  ))}
                </ul>

                <h3 style={{ marginTop: 24 }}>Risk Metrics:</h3>
                <p>Expected Return: {result.performance_metrics.expected_annual_return}%</p>
                <p>Volatility: {result.performance_metrics.expected_annual_volatility}%</p>
                <p>Sharpe Ratio: {result.performance_metrics.sharpe_ratio}</p>
                <p>Max Drawdown: {result.performance_metrics.max_drawdown}</p>

                <h3 style={{ marginTop: 24 }}>Backtest Chart:</h3>
                <img
                  src={`data:image/png;base64,${result.backtest_chart}`}
                  alt="backtest"
                  style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid #000' }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
