import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function StockBasic() {
  const [ticker, setTicker] = useState('');
  const navigate = useNavigate();

  const handleSearch = () => {
    if (ticker) {
      navigate(`/stock/${ticker}`);
    }
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
        minWidth: 320,
        textAlign: 'center'
      }}>
        <h2 style={{marginBottom: 24, color: '#4f46e5'}}>股票基础信息查询</h2>
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="请输入股票代码（如 AAPL）"
          style={{
            padding: '10px 16px',
            border: '1px solid #c7d2fe',
            borderRadius: 6,
            fontSize: 16,
            outline: 'none',
            width: '80%',
            marginBottom: 20
          }}
          onKeyDown={e => { if (e.key === 'Enter') handleSearch(); }}
        />
        <br />
        <button
          onClick={handleSearch}
          style={{
            background: '#6366f1',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '10px 32px',
            fontSize: 16,
            cursor: 'pointer',
            transition: 'background 0.2s'
          }}
        >
          查询
        </button>
        <div style={{marginTop: 24, color: '#64748b', fontSize: 14}}>
          示例：AAPL、TSLA、MSFT、GOOG
        </div>
      </div>
    </div>
  );
}