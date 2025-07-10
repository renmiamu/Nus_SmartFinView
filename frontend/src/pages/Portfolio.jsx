// Portfolio.jsx – Black & White Theme with NavBar
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { useNavigate, useLocation } from 'react-router-dom';

const GRAY_COLORS = ['#000', '#333', '#555', '#777', '#999', '#bbb'];

export default function Portfolio() {
  const navigate = useNavigate();
  const location = useLocation();
  const menuItems = [
    { label: 'Basic Info',      path: '/stock' },
    { label: 'Portfolio Viz',   path: '/portfolio' },
    { label: 'AI Score',        path: '/score' },
    { label: 'Sentiment',       path: '/emotion' },
    { label: 'Recommendations', path: '/recommendation' }
  ];

  const [holdings, setHoldings] = useState([{ ticker: '', shares: '', buyPrice: '' }]);
  const [result, setResult] = useState(null);
  const [navVisible, setNavVisible] = useState(false);

  // Nav animation
  useEffect(() => {
    const timer = setTimeout(() => setNavVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const addRow = () => setHoldings([...holdings, { ticker: '', shares: '', buyPrice: '' }]);
  const removeRow = idx => setHoldings(holdings.filter((_, i) => i !== idx));
  const updateRow = (idx, key, value) => {
    const copy = [...holdings]; copy[idx][key] = value; setHoldings(copy);
  };

  const handleCalculate = async () => {
    const payload = holdings
      .filter(h => h.ticker && h.shares && h.buyPrice)
      .map(h => ({ ticker: h.ticker, shares: +h.shares, buy_price: +h.buyPrice }));
    try {
      const res = await axios.post('http://localhost:8000/stock/profit/batch', payload);
      setResult(res.data);
    } catch {
      alert('Calculation failed.');
    }
  };

  const pieData = result?.map(r => ({ name: r.ticker, value: r.current_value })) || [];
  const barData = result?.map(r => ({ name: r.ticker, profit: r.profit })) || [];
  const totalCost  = result?.reduce((s, r) => s + r.cost, 0) || 0;
  const totalValue = result?.reduce((s, r) => s + r.current_value, 0) || 0;
  const totalReturn = totalCost ? ((totalValue - totalCost) / totalCost * 100).toFixed(2) : 0;

  return (
    <div style={{ background: '#fff', minHeight: '100vh', color: '#000', overflow: 'hidden' }}>
      {/* Navbar */}
      <style>{`
        @keyframes fadeInUp { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
        @keyframes slideInRight { from {transform:translateX(100px);opacity:0;} to {transform:translateX(0);opacity:1;} }
        .navbar { position:fixed;top:0;width:100%;background:#000;display:flex;justify-content:center;padding:16px 0;z-index:1000; }
        .nav-item { color:#777;margin:0 24px;font-size:16px;font-weight:500;position:relative;padding-bottom:6px;cursor:pointer;opacity:0;transition:color .3s; }
        .nav-item:hover, .nav-item.active { color:#fff; }
        .nav-item.active::after { content:'';position:absolute;bottom:0;left:0;width:100%;height:2px;background:#fff; }
      `}</style>
      <nav className="navbar">
        {menuItems.map((item, idx) => (
          <div
            key={item.path}
            onClick={() => navigate(item.path)}
            className={`nav-item${location.pathname===item.path?' active':''}`}
            style={{ animation: navVisible ? `slideInRight .6s ease-out forwards ${idx*0.15}s` : 'none' }}
          >{item.label}</div>
        ))}
      </nav>

      <div style={{ padding: '100px 20px 40px', maxWidth: 1000, margin: '0 auto', fontFamily:'Helvetica, Arial, sans-serif' }}>
        <h2 style={{ marginBottom: 24, textAlign:'center' }}>Portfolio Analytics</h2>

        {/* Input Table */}
        <table style={{ width:'100%', borderCollapse:'collapse', marginBottom:24 }}>
          <thead>
            <tr style={{ background:'#f9f9f9', color:'#000', fontWeight:600 }}>
              <th style={{ padding:12, border:'1px solid #e5e5e5' }}>Ticker</th>
              <th style={{ padding:12, border:'1px solid #e5e5e5' }}>Buy Price</th>
              <th style={{ padding:12, border:'1px solid #e5e5e5' }}>Shares</th>
              <th style={{ padding:12, border:'1px solid #e5e5e5' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {holdings.map((h, i) => (
              <tr key={i} style={{ textAlign:'center', borderTop:'1px solid #e5e5e5' }}>
                {['ticker','buyPrice','shares'].map(f => (
                  <td key={f} style={{ padding:8 }}>
                    <input
                      value={h[f]}
                      onChange={e=>updateRow(i,f,e.target.value)}
                      style={{ width:'90%', padding:6, border:'1px solid #ccc', borderRadius:4, outline:'none' }}
                    />
                  </td>
                ))}
                <td style={{ padding:8 }}>
                  {holdings.length>1 && (
                    <button onClick={()=>removeRow(i)} style={{ background:'none',border:'none',color:'#f00',cursor:'pointer' }}>Remove</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ textAlign:'center', marginBottom:40 }}>
          <button onClick={addRow} style={{ marginRight:16, padding:'8px 16px', border:'1px solid #000', background:'#fff', cursor:'pointer' }}>Add Row</button>
          <button onClick={handleCalculate} style={{ padding:'8px 32px', border:'none', background:'#000', color:'#fff', cursor:'pointer' }}>Calculate</button>
        </div>

        {/* Results */}
        {result && (
          <>
            <div style={{ textAlign:'center', marginBottom:24 }}>
              <span>Total Return: </span>
              <span style={{ fontSize:20, fontWeight:600, color: totalReturn>=0?'#000':'#000' }}>{totalReturn}%</span>
            </div>
            <div style={{ marginBottom:32 }}>
              <ResponsiveContainer width='100%' height={300}>
                <PieChart>
                  <Pie data={pieData} dataKey='value' nameKey='name' outerRadius={100} label>
                    {pieData.map((e,idx)=><Cell key={e.name} fill={GRAY_COLORS[idx%GRAY_COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div>
              <ResponsiveContainer width='100%' height={300}>
                <BarChart data={barData}>
                  <XAxis dataKey='name' tick={{ fill:'#000' }} />
                  <YAxis tick={{ fill:'#000' }} />
                  <Tooltip />
                  <Bar dataKey='profit'>
                    {barData.map((e,idx)=><Cell key={e.name} fill={GRAY_COLORS[idx%GRAY_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
