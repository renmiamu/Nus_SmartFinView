import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer
} from 'recharts';

export default function StockDetail() {
  const navigate = useNavigate();
  const location = useLocation();
  const { ticker } = useParams();
  const menuItems = [
    { label: 'Basic Info',       path: '/stock' },
    { label: 'Portfolio Viz',    path: '/portfolio' },
    { label: 'AI Score',         path: '/score' },
    { label: 'Sentiment',        path: '/emotion' },
    { label: 'Recommendations',  path: '/recommendation' }
  ];

  const [animateNav, setAnimateNav] = useState(false);
  const [data, setData] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => setAnimateNav(true), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    axios.get(`http://localhost:8000/stock/basic?ticker=${ticker}`)
      .then(res => setData(res.data));
  }, [ticker]);

  const commonContainer = {
    background: '#fff',
    padding: '40px 0',
    minHeight: 'calc(100vh - 80px)'
  };

  return (
    <div style={{
      position: 'relative',
      minHeight: '100vh',
      background: '#fff',
      color: '#000',
      overflow: 'hidden'
    }}>
      {/* Navbar styles */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to   { transform: translateX(0); opacity: 1; }
        }
        .navbar {
          position: fixed;
          top: 0;
          width: 100%;
          background: #111;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          display: flex;
          justify-content: center;
          padding: 16px 0;
          z-index: 1000;
        }
        .nav-item {
          color: #aaa;
          margin: 0 24px;
          font-size: 16px;
          font-weight: 500;
          position: relative;
          padding-bottom: 6px;
          transition: color 0.3s;
          cursor: pointer;
          opacity: 0;
        }
        .nav-item:hover { color: #fff; }
        .nav-item.active { color: #fff; }
        .nav-item.active::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          width: 100%;
          height: 2px;
          background: #fff;
        }
      `}</style>

      {/* Top navigation */}
      <nav className="navbar">
        {menuItems.map((item, idx) => {
          const isActive = location.pathname === item.path;
          return (
            <div
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`nav-item${isActive ? ' active' : ''}`}
              style={{
                animation: animateNav
                  ? `slideInRight 0.6s ease-out forwards ${idx * 0.15}s`
                  : 'none'
              }}
            >
              {item.label}
            </div>
          );
        })}
      </nav>

      {/* Content below navbar */}
      <div style={commonContainer}>
        {!data ? (
          <div style={{ textAlign: 'center', padding: 40, color: '#000' }}>加载中...</div>
        ) : data.error ? (
          <div style={{ textAlign: 'center', padding: 40, color: '#000' }}>{data.error}</div>
        ) : (
          <div style={{
            maxWidth: 1000,
            margin: '0 auto',
            background: '#f9f9f9',
            borderRadius: 18,
            border: '1px solid #ddd',
            boxShadow: '0 4px 24px rgba(0,0,0,0.1)',
            padding: '40px 40px 32px 40px'
          }}>
            <h2 style={{ color: '#000', marginBottom: 32, fontWeight: 700, letterSpacing: 1 }}>
              {ticker} 股票详情
            </h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 32, marginBottom: 32, color: '#000', fontSize: 18 }}>
              <div><span style={{ color: '#666' }}>最新价格：</span><span style={{ fontWeight: 600 }}>{data.latest_price}</span></div>
              <div><span style={{ color: '#666' }}>涨跌幅：</span><span style={{ fontWeight: 600 }}>{data.change_percent.toFixed(2)}%</span></div>
              <div><span style={{ color: '#666' }}>最新成交量：</span><span style={{ fontWeight: 600 }}>{data.volume}</span></div>
              <div><span style={{ color: '#666' }}>今日总成交量：</span><span style={{ fontWeight: 600 }}>{data.total_volume}</span></div>
            </div>
            <h3 style={{ color: '#000', marginBottom: 16, fontWeight: 500 }}>价格变化趋势</h3>
            <div style={{ background: '#fff', borderRadius: 12, padding: 16, border: '1px solid #ddd', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={data.time_series.map((t, i) => ({ time: t, price: data.close_series[i] }))}>
                  <CartesianGrid stroke="#ccc" strokeDasharray="3 3" />
                  <XAxis dataKey="time" minTickGap={40} tick={{ fontSize: 12, fill: '#000' }} />
                  <YAxis domain={[ 'auto', 'auto' ]} tick={{ fontSize: 12, fill: '#000' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }} />
                  <Line type="monotone" dataKey="price" stroke="#000" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
