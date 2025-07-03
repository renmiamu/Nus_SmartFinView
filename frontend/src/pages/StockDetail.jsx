// filepath: /Users/hlshen/Desktop/Nus_SmartFinView/frontend/src/pages/StockDetail.jsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

export default function StockDetail() {
  const { ticker } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get(`http://localhost:8000/stock/basic?ticker=${ticker}`).then(res => {
      setData(res.data);
    });
  }, [ticker]);

  if (!data) return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)'
    }}>
      <div style={{
        background: '#fff',
        padding: 40,
        borderRadius: 16,
        boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
        minWidth: 320,
        textAlign: 'center',
        fontSize: 18,
        color: '#6366f1'
      }}>
        加载中...
      </div>
    </div>
  );
  if (data.error) return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)'
    }}>
      <div style={{
        background: '#fff',
        padding: 40,
        borderRadius: 16,
        boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
        minWidth: 320,
        textAlign: 'center',
        fontSize: 18,
        color: '#ef4444'
      }}>
        {data.error}
      </div>
    </div>
  );

  const chartData = data.time_series.map((time, idx) => ({
    time,
    price: data.close_series[idx]
  }));

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)',
      padding: '40px 0'
    }}>
      <div style={{
        maxWidth: 1000,
        margin: '0 auto',
        background: '#fff',
        borderRadius: 18,
        boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
        padding: '40px 40px 32px 40px'
      }}>
        <h2 style={{ color: '#4f46e5', marginBottom: 32, fontWeight: 700, letterSpacing: 1 }}>
          {ticker} 股票详情
        </h2>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 32,
          marginBottom: 32
        }}>
          <div style={{ fontSize: 18, color: '#334155' }}>
            <span style={{ color: '#64748b' }}>最新价格：</span>
            <span style={{ color: '#16a34a', fontWeight: 600 }}>{data.latest_price}</span>
          </div>
          <div style={{ fontSize: 18, color: '#334155' }}>
            <span style={{ color: '#64748b' }}>涨跌幅：</span>
            <span style={{ color: data.change_percent >= 0 ? '#ef4444' : '#22d3ee', fontWeight: 600 }}>
              {data.change_percent.toFixed(2)}%
            </span>
          </div>
          <div style={{ fontSize: 18, color: '#334155' }}>
            <span style={{ color: '#64748b' }}>最新成交量：</span>
            <span style={{ fontWeight: 600 }}>{data.volume}</span>
          </div>
          <div style={{ fontSize: 18, color: '#334155' }}>
            <span style={{ color: '#64748b' }}>今日总成交量：</span>
            <span style={{ fontWeight: 600 }}>{data.total_volume}</span>
          </div>
        </div>
        <h3 style={{ color: '#4f46e5', marginBottom: 16, fontWeight: 500 }}>价格变化趋势</h3>
        <div style={{
          background: '#f8fafc',
          borderRadius: 12,
          padding: 16,
          boxShadow: '0 2px 8px rgba(0,0,0,0.03)'
        }}>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
              <XAxis dataKey="time" minTickGap={40} tick={{ fontSize: 12, fill: '#64748b' }} />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12, fill: '#64748b' }} />
              <Tooltip />
              <Line type="monotone" dataKey="price" stroke="#6366f1" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}