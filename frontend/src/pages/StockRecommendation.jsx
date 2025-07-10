import React, { useState } from 'react';
import axios from 'axios';

export default function StockRecommendation() {
  const [form, setForm] = useState({
    risk_tolerance: 'medium',
    industry_preference: [],
    investment_amount: '',
    lookback_period: 365
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const industries = ['Technology', 'Healthcare', 'Finance', 'Energy'];

  const handleSubmit = async () => {
    if (!form.investment_amount) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post('http://localhost:8000/stock/recommendation', form);
      setResult(res.data);
    } catch (e) {
      setResult({ error: e.response?.data?.detail || '请求失败' });
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
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)',
      padding: 40
    }}>
      <div style={{
        background: '#fff',
        padding: 40,
        borderRadius: 16,
        boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
        maxWidth: 720,
        width: '100%'
      }}>
        <h2 style={{ color: '#14b8a6', marginBottom: 24 }}>智能投资组合推荐</h2>

        <div style={{ marginBottom: 16 }}>
          <label>投资金额（美元）：</label><br />
          <input
            type="number"
            value={form.investment_amount}
            onChange={e => setForm({ ...form, investment_amount: parseFloat(e.target.value) || '' })}
            style={{ padding: 8, borderRadius: 6, border: '1px solid #ccc', width: '60%' }}
          />
        </div>

        <div style={{ marginBottom: 16 }}>
          <label>风险偏好：</label><br />
          <select
            value={form.risk_tolerance}
            onChange={e => setForm({ ...form, risk_tolerance: e.target.value })}
            style={{ padding: 8, borderRadius: 6, border: '1px solid #ccc' }}
          >
            <option value="low">低</option>
            <option value="medium">中</option>
            <option value="high">高</option>
          </select>
        </div>

        <div style={{ marginBottom: 16 }}>
          <label>行业偏好：</label><br />
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

        <button
          onClick={handleSubmit}
          disabled={loading}
          style={{
            background: '#14b8a6',
            color: '#fff',
            padding: '10px 24px',
            borderRadius: 6,
            fontSize: 16,
            border: 'none',
            cursor: 'pointer'
          }}
        >
          {loading ? '生成中...' : '生成推荐'}
        </button>

        {result && (
          <div style={{ marginTop: 32 }}>
            {result.error ? (
              <div style={{ color: '#ef4444' }}>{result.error}</div>
            ) : (
              <div>
                <h3 style={{ color: '#334155', marginBottom: 12 }}>推荐资产及权重：</h3>
                <ul>
                  {result.recommended_assets.map(({ ticker, weight }, idx) => (
                    <li key={idx}>{ticker} - {weight}%</li>
                  ))}
                </ul>

                <h3 style={{ color: '#334155', marginTop: 24 }}>风险指标：</h3>
                <p>预期年化收益率：{result.performance_metrics.expected_annual_return}%</p>
                <p>年化波动率：{result.performance_metrics.expected_annual_volatility}%</p>
                <p>夏普比率：{result.performance_metrics.sharpe_ratio}</p>
                <p>最大回撤：{result.performance_metrics.max_drawdown}</p>

                <h3 style={{ color: '#334155', marginTop: 24 }}>回测图表：</h3>
                <img
                  src={`data:image/png;base64,${result.backtest_chart}`}
                  alt="backtest chart"
                  style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid #ccc' }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
