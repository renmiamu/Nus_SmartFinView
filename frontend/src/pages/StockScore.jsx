import React, { useState } from 'react';
import axios from 'axios';

export default function StockScore() {
  const [ticker, setTicker] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleScore = async () => {
    if (!ticker) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.get('http://localhost:8000/stock/score', {
        params: { ticker }
      });
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.response?.data?.detail || '查询失败' });
    }
    setLoading(false);
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)'
    }}>
      <div style={{
        background: '#fff',
        padding: '40px 32px',
        borderRadius: '16px',
        boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
        minWidth: 340,
        textAlign: 'center'
      }}>
        <h2 style={{marginBottom: 24, color: '#0ea5e9'}}>智能打分</h2>
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="请输入股票代码（如 AAPL）"
          style={{
            padding: '10px 16px',
            border: '1px solid #bae6fd',
            borderRadius: 6,
            fontSize: 16,
            outline: 'none',
            width: '80%',
            marginBottom: 20
          }}
          onKeyDown={e => { if (e.key === 'Enter') handleScore(); }}
        />
        <br />
        <button
          onClick={handleScore}
          style={{
            background: '#0ea5e9',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '10px 32px',
            fontSize: 16,
            cursor: 'pointer',
            marginBottom: 20
          }}
          disabled={loading}
        >
          {loading ? '查询中...' : '智能打分'}
        </button>
        {result && (
          <div style={{marginTop: 24, color: result.error ? '#ef4444' : '#0ea5e9', fontSize: 18}}>
            {result.error
              ? result.error
              : (
                <div>
                  <div>股票代码：{result.ticker}</div>
                  <div>模型分数：<span style={{fontWeight: 600}}>{result.score}</span></div>
                </div>
              )
            }
          </div>
        )}
        <div style={{marginTop: 24, color: '#64748b', fontSize: 14}}>
          示例：AAPL、TSLA、MSFT、GOOG
        </div>
      </div>
    </div>
  );
}