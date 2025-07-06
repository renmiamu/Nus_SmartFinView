import React, { useState } from 'react';
import axios from 'axios';
import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#6366f1'];

export default function Portfolio() {
  const [holdings, setHoldings] = useState([{ ticker: '', shares: '', buyPrice: '' }]);
  const [result, setResult] = useState(null);

  const addRow = () => setHoldings([...holdings, { ticker: '', shares: '', buyPrice: '' }]);
  const removeRow = idx => setHoldings(holdings.filter((_, i) => i !== idx));
  const updateRow = (idx, key, value) => {
    const newHoldings = [...holdings];
    newHoldings[idx][key] = value;
    setHoldings(newHoldings);
  };

  const handleCalculate = async () => {
    const payload = holdings
      .filter(h => h.ticker && h.shares && h.buyPrice)
      .map(h => ({
        ticker: h.ticker,
        shares: parseFloat(h.shares),
        buy_price: parseFloat(h.buyPrice)
      }));

    try {
      const res = await axios.post('http://localhost:8000/stock/profit/batch', payload);
      setResult(res.data);
    } catch (err) {
      console.error('请求失败', err);
      alert("查询失败，请检查股票代码或稍后再试");
    }
  };

  const pieData = result?.map(r => ({ name: r.ticker, value: r.current_value })) || [];
  const barData = result?.map(r => ({ name: r.ticker, profit: r.profit })) || [];

  const totalCost = result?.reduce((sum, r) => sum + r.cost, 0) || 0;
  const totalValue = result?.reduce((sum, r) => sum + r.current_value, 0) || 0;
  const totalReturn = totalCost ? ((totalValue - totalCost) / totalCost * 100).toFixed(2) : 0;

  return (
    <div style={{ background: '#f3f4f6', minHeight: '100vh', padding: '40px 20px' }}>
      <div style={{
        maxWidth: 1000,
        margin: '0 auto',
        background: '#f9fafb',
        borderRadius: 16,
        padding: 40,
        boxShadow: '0 8px 30px rgba(0,0,0,0.05)',
        fontFamily: 'Arial, sans-serif'
      }}>
        <h2 style={{ color: '#1e40af', marginBottom: 32, fontWeight: 'bold' }}>投资组合收益可视化</h2>

        {/* 表格输入 */}
        <table style={{
          width: '100%',
          marginBottom: 24,
          borderRadius: 12,
          borderCollapse: 'collapse',
          overflow: 'hidden',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.06)'
        }}>
          <thead>
            <tr style={{ backgroundColor: '#e0f2fe', color: '#1e3a8a', fontWeight: 600 }}>
              <th style={{ padding: '12px 8px', borderRight: '1px solid #e5e7eb' }}>股票代码</th>
              <th style={{ padding: '12px 8px', borderRight: '1px solid #e5e7eb' }}>买入价</th>
              <th style={{ padding: '12px 8px', borderRight: '1px solid #e5e7eb' }}>持仓数量</th>
              <th style={{ padding: '12px 8px' }}>操作</th>
            </tr>
          </thead>
          <tbody style={{ backgroundColor: '#f3f4f6' }}>
            {holdings.map((h, idx) => (
              <tr key={idx} style={{ textAlign: 'center', borderTop: '1px solid #e5e7eb' }}>
                {['ticker', 'buyPrice', 'shares'].map(field => (
                  <td style={{ padding: 10 }} key={field}>
                    <input
                      value={h[field]}
                      onChange={e => updateRow(idx, field, e.target.value)}
                      style={{
                        width: '90%',
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid #d1d5db',
                        background: '#e0f2fe',
                        outline: 'none',
                        fontSize: 14
                      }}
                      onFocus={e => e.target.style.border = '1px solid #3b82f6'}
                      onBlur={e => e.target.style.border = '1px solid #d1d5db'}
                    />
                  </td>
                ))}
                <td style={{ padding: 10 }}>
                  {holdings.length > 1 && (
                    <button
                      onClick={() => removeRow(idx)}
                      style={{
                        color: '#dc2626',
                        border: 'none',
                        background: 'none',
                        cursor: 'pointer',
                        fontSize: 14
                      }}
                      onMouseEnter={e => e.target.style.color = '#b91c1c'}
                      onMouseLeave={e => e.target.style.color = '#dc2626'}
                    >
                      删除
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* 操作按钮 */}
        <div style={{ marginBottom: 40 }}>
          <button onClick={addRow} style={{
            marginRight: 16,
            background: '#c7d2fe',
            color: '#1e3a8a',
            border: 'none',
            borderRadius: 6,
            padding: '8px 16px',
            fontSize: 14,
            cursor: 'pointer'
          }}>添加一行</button>

          <button onClick={handleCalculate} style={{
            background: '#2563eb',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '8px 32px',
            fontSize: 16,
            cursor: 'pointer'
          }}>计算收益</button>
        </div>

        {/* 图表展示 */}
        {result && (
          <div>
            <div style={{ fontSize: 18, marginBottom: 24 }}>
              当前总收益率：
              <span style={{ color: totalReturn >= 0 ? '#16a34a' : '#dc2626', fontWeight: 600 }}>
                {totalReturn}%
              </span>
            </div>

            {/* 饼图区域 */}
            <div style={{ background: '#f0f4f8', borderRadius: 12, padding: 20, marginBottom: 32 }}>
              <h3 style={{ color: '#0369a1', marginBottom: 16 }}>持仓分布</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={100} label>
                    {pieData.map((entry, idx) => (
                      <Cell key={entry.name} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* 柱状图区域 */}
            <div style={{ background: '#f0f4f8', borderRadius: 12, padding: 20 }}>
              <h3 style={{ color: '#0369a1', marginBottom: 16 }}>个股盈亏</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={barData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="profit">
                    {barData.map((entry, idx) => (
                      <Cell key={`bar-${entry.name}`} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
