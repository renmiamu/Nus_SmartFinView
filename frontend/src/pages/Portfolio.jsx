import React, { useState } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#6366f1', '#22d3ee', '#f59e42', '#ef4444', '#16a34a', '#a21caf'];

export default function Portfolio() {
  const [holdings, setHoldings] = useState([
    { ticker: '', shares: '', buyPrice: '' }
  ]);
  const [result, setResult] = useState(null);

  // 添加一行
  const addRow = () => setHoldings([...holdings, { ticker: '', shares: '', buyPrice: '' }]);
  // 删除一行
  const removeRow = idx => setHoldings(holdings.filter((_, i) => i !== idx));
  // 修改输入
  const updateRow = (idx, key, value) => {
    const newHoldings = [...holdings];
    newHoldings[idx][key] = value;
    setHoldings(newHoldings);
  };

  // 计算收益
  const handleCalculate = async () => {
    const responses = await Promise.all(
      holdings
        .filter(h => h.ticker && h.shares && h.buyPrice)
        .map(h =>
          axios.get('http://localhost:8000/stock/profit', {
            params: { ticker: h.ticker, shares: h.shares, buy_price: h.buyPrice }
          })
        )
    );
    setResult(responses.map(r => r.data));
  };

  // 饼图数据
  const pieData = result?.map(r => ({
    name: r.ticker,
    value: r.current_value
  })) || [];

  // 柱状图数据
  const barData = result?.map(r => ({
    name: r.ticker,
    profit: r.profit
  })) || [];

  // 总收益率
  const totalCost = result?.reduce((sum, r) => sum + r.cost, 0) || 0;
  const totalValue = result?.reduce((sum, r) => sum + r.current_value, 0) || 0;
  const totalReturn = totalCost ? ((totalValue - totalCost) / totalCost * 100).toFixed(2) : 0;

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: 40 }}>
      <h2 style={{ color: '#4f46e5', marginBottom: 32 }}>投资组合收益可视化</h2>
      <table style={{ width: '100%', marginBottom: 24, background: '#fff', borderRadius: 12, boxShadow: '0 2px 8px rgba(0,0,0,0.03)' }}>
        <thead>
          <tr style={{ background: '#f1f5f9' }}>
            <th>股票代码</th>
            <th>买入价</th>
            <th>持仓数量</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h, idx) => (
            <tr key={idx}>
              <td>
                <input value={h.ticker} onChange={e => updateRow(idx, 'ticker', e.target.value.toUpperCase())} style={{ width: 80 }} />
              </td>
              <td>
                <input value={h.buyPrice} onChange={e => updateRow(idx, 'buyPrice', e.target.value)} style={{ width: 80 }} />
              </td>
              <td>
                <input value={h.shares} onChange={e => updateRow(idx, 'shares', e.target.value)} style={{ width: 80 }} />
              </td>
              <td>
                {holdings.length > 1 && <button onClick={() => removeRow(idx)} style={{ color: '#ef4444', border: 'none', background: 'none', cursor: 'pointer' }}>删除</button>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={addRow} style={{ marginRight: 16, background: '#e0e7ff', border: 'none', borderRadius: 6, padding: '8px 16px', cursor: 'pointer' }}>添加一行</button>
      <button onClick={handleCalculate} style={{ background: '#6366f1', color: '#fff', border: 'none', borderRadius: 6, padding: '8px 32px', fontSize: 16, cursor: 'pointer' }}>计算收益</button>
      {result && (
        <div style={{ marginTop: 40 }}>
          <div style={{ fontSize: 18, marginBottom: 24 }}>
            当前总收益率：<span style={{ color: totalReturn >= 0 ? '#ef4444' : '#22d3ee', fontWeight: 600 }}>{totalReturn}%</span>
          </div>
          <h3 style={{ color: '#4f46e5', marginBottom: 16 }}>持仓分布</h3>
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
          <h3 style={{ color: '#4f46e5', margin: '32px 0 16px' }}>个股盈亏</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="profit" fill="#6366f1" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}