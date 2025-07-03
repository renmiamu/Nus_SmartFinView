// filepath: /Users/hlshen/Desktop/Nus_SmartFinView/frontend/src/pages/StockDetail.jsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

export default function StockDetail() {
  const { ticker } = useParams();
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get(`http://localhost:8000/stock/basic?ticker=${ticker}`).then(res => {
      setData(res.data);
    });
  }, [ticker]);

  if (!data) return <div>加载中...</div>;
  if (data.error) return <div style={{color: 'red'}}>{data.error}</div>;

  return (
    <div>
      <h2>{ticker} 股票详情</h2>
      <div>最新价格: {data.latest_price}</div>
      <div>涨跌幅: {data.change_percent.toFixed(2)}%</div>
      <div>最新成交量: {data.volume}</div>
      <div>今日总成交量: {data.total_volume}</div>
    </div>
  );
}