import React, { useState } from 'react';
import axios from 'axios';

export default function StockBasic() {
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState(null);

  const fetchData = async () => {
    const res = await axios.get(`http://localhost:8000/stock/basic?ticker=${ticker}`);
    setData(res.data);
  };

  return (
    <div>
      <input value={ticker} onChange={e => setTicker(e.target.value)} placeholder="输入股票代码" />
      <button onClick={fetchData}>查询</button>
      {data && data.error && <div style={{color: 'red'}}>{data.error}</div>}
      {data && !data.error && (
        <div>
          <div>最新价格: {data.latest_price}</div>
          <div>涨跌幅: {data.change_percent.toFixed(2)}%</div>
          <div>最新成交量: {data.volume}</div>
          <div>今日总成交量: {data.total_volume}</div>
          {/* 这里可以用 ECharts、Recharts 等库绘制 data.close_series 和 data.time_series */}
        </div>
      )}
    </div>
  );
}