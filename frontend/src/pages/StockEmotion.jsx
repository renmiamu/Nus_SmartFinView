import React, { useState } from 'react';
import axios from 'axios';

export default function StockEmotion() {
  const [keyword, setKeyword] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleEmotion = async () => {
    if (!keyword) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.get('http://localhost:8000/stock/emotion', {
        params: { keyword }
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
        minWidth: 360,
        textAlign: 'center'
      }}>
        <h2 style={{ marginBottom: 24, color: '#f97316' }}>舆情情绪分析</h2>
        <input
          value={keyword}
          onChange={e => setKeyword(e.target.value)}
          placeholder="请输入关键词（如 Tesla）"
          style={{
            padding: '10px 16px',
            border: '1px solid #fcd34d',
            borderRadius: 6,
            fontSize: 16,
            outline: 'none',
            width: '80%',
            marginBottom: 20
          }}
          onKeyDown={e => { if (e.key === 'Enter') handleEmotion(); }}
        />
        <br />
        <button
          onClick={handleEmotion}
          style={{
            background: '#f97316',
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
          {loading ? '分析中...' : '分析情绪'}
        </button>
        {result && (
          <div style={{ marginTop: 24, fontSize: 16, color: result.error ? '#ef4444' : '#374151' }}>
            {result.error
              ? result.error
              : (
                <div>
                  <div><strong>关键词：</strong>{result.keyword}</div>
                  <div><strong>平均情绪分：</strong>{result.avg_compound}</div>
                  <div><strong>情绪等级：</strong>{result.emotion_level}</div>
                  <div><strong>建议：</strong><span style={{ color: '#f97316' }}>{result.suggestion}</span></div>
                  <div style={{ marginTop: 16, textAlign: 'left' }}>
                    <strong>高频词：</strong>
                    <ul style={{ paddingLeft: 20 }}>
                      {result.top_words.slice(0, 10).map(({ word, count }, idx) => (
                        <li key={idx}>{word}（{count}）</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )
            }
          </div>
        )}
      </div>
    </div>
  );
}
