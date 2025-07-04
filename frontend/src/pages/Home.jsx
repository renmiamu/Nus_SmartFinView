import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)'
    }}>
      <div style={{
        display: 'flex',
        gap: 48,
        flexWrap: 'wrap'
      }}>
        <div
          onClick={() => navigate('/stock')}
          style={{
            background: '#fff',
            padding: '48px 40px',
            borderRadius: 18,
            boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
            minWidth: 320,
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'box-shadow 0.2s',
            border: '2px solid #6366f1'
          }}
        >
          <h2 style={{ color: '#4f46e5', marginBottom: 24 }}>股票基础信息查询</h2>
          <div style={{ color: '#64748b', fontSize: 16, marginBottom: 12 }}>
            输入股票代码，查询实时行情与走势
          </div>
          <button
            style={{
              background: '#6366f1',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '10px 32px',
              fontSize: 16,
              cursor: 'pointer'
            }}
            onClick={e => { e.stopPropagation(); navigate('/stock'); }}
          >
            进入查询
          </button>
        </div>
        <div
          onClick={() => navigate('/portfolio')}
          style={{
            background: '#fff',
            padding: '48px 40px',
            borderRadius: 18,
            boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
            minWidth: 320,
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'box-shadow 0.2s',
            border: '2px solid #22d3ee'
          }}
        >
          <h2 style={{ color: '#0ea5e9', marginBottom: 24 }}>投资组合收益可视化</h2>
          <div style={{ color: '#64748b', fontSize: 16, marginBottom: 12 }}>
            输入持仓，查看收益分布与盈亏分析
          </div>
          <button
            style={{
              background: '#22d3ee',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              padding: '10px 32px',
              fontSize: 16,
              cursor: 'pointer'
            }}
            onClick={e => { e.stopPropagation(); navigate('/portfolio'); }}
          >
            进入组合
          </button>
        </div>
      </div>
    </div>
  );
}