// StockEmotion.jsx – Polished Black & White Layout with NavBar
import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer
} from "recharts";
import ReactECharts from 'echarts-for-react';
import 'echarts-wordcloud';
import { useNavigate, useLocation } from 'react-router-dom';

export default function StockEmotion() {
  const navigate = useNavigate();
  const location = useLocation();
  const menuItems = [
    { label: 'Basic Info', path: '/stock' },
    { label: 'Portfolio Viz', path: '/portfolio' },
    { label: 'AI Score', path: '/score' },
    { label: 'Sentiment', path: '/emotion' },
    { label: 'Recommendations', path: '/recommendation' }
  ];

  const [keyword, setKeyword] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [wordCloudData, setWordCloudData] = useState([]);
  const [showTrend, setShowTrend] = useState(false);
  const [showWordCloud, setShowWordCloud] = useState(false);
  const [navVisible, setNavVisible] = useState(false);
  const [trendType, setTrendType] = useState('compound'); // compound/compound_vader/compound_bert

  useEffect(() => {
    const t = setTimeout(() => setNavVisible(true), 300);
    return () => clearTimeout(t);
  }, []);

  const handleEmotion = async () => {
    if (!keyword) return;
    setLoading(true);
    setResult(null);
    try {
      const { data } = await axios.get('/stock/emotion', { params: { keyword } });
      setResult(data);
      const news = data.news_items || [];
      setTrendData(news.filter(i => i.publishedAt).map(i => ({
        time: i.publishedAt.slice(0, 16).replace('T', ' '),
        compound: i.compound !== undefined ? i.compound : (i.compound_vader + i.compound_bert) / 2,
        compound_vader: i.compound_vader,
        compound_bert: i.compound_bert
      })));
      setWordCloudData((data.top_words || []).map(w => ({ name: w.word, value: w.count })));
    } catch (e) {
      setResult({ error: e.response?.data?.detail || 'Query failed' });
      setTrendData([]);
      setWordCloudData([]);
    }
    setLoading(false);
  };

  return (
    <div style={{ background: '#fff', color: '#000', minHeight: '100vh', overflow: 'hidden' }}>
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .navbar { position: fixed; top: 0; width: 100%; background: #000; display: flex; justify-content: center; padding: 12px 0; z-index: 1000; }
        .nav-item { color: #777; margin: 0 20px; font-size: 14px; font-weight: 500; opacity: 0; position: relative; padding-bottom: 4px; cursor: pointer; transition: color .3s; }
        .nav-item:hover, .nav-item.active { color: #fff; }
        .nav-item.active::after { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background: #fff; }
        .btn { background: #000; color: #fff; border: none; border-radius: 6px; padding: 8px 20px; font-size: 14px; cursor: pointer; }
        .section { margin: 48px 0; }
        .card { background: #f4f4f4; padding: 20px 24px; border-radius: 8px; max-width: 640px; margin: 24px auto; text-align: left; line-height: 1.6; }
      `}</style>

      <nav className="navbar">
        {menuItems.map((item, idx) => (
          <div
            key={item.path}
            onClick={() => navigate(item.path)}
            className={`nav-item${location.pathname === item.path ? ' active' : ''}`}
            style={{ animation: navVisible ? `slideInRight 0.5s ease-out forwards ${idx * 0.1}s` : 'none' }}
          >{item.label}</div>
        ))}
      </nav>

      <div style={{ paddingTop: 80, maxWidth: 860, margin: '0 auto', padding: 20 }}>
        {/* Input Section */}
        <div className="section" style={{ textAlign: 'center' }}>
          <h2>Sentiment Analysis</h2>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginTop: 16 }}>
            <input
              value={keyword}
              onChange={e => setKeyword(e.target.value)}
              placeholder="Enter keyword (e.g. Tesla)"
              style={{ padding: '8px 12px', width: 260, border: '1px solid #000', borderRadius: 4, fontSize: 14 }}
              onKeyDown={e => e.key === 'Enter' && handleEmotion()}
            />
            <button className="btn" onClick={handleEmotion} disabled={loading}>
              {loading ? 'Loading...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Info Card */}
        {result && (
          <div className="card">
            {result.error ? (
              <p style={{ color: '#800' }}>{result.error}</p>
            ) : (
              <>
                <p><strong>Keyword:</strong> {result.keyword}</p>
                <p><strong>Average Score (Fusion):</strong> {result.avg_compound}</p>
                <p><strong>Average VADER:</strong> {result.avg_compound_vader}</p>
                <p><strong>Average BERT:</strong> {result.avg_compound_bert}</p>
                <p><strong>Level:</strong> {result.emotion_level}</p>
                <p><strong>Suggestion:</strong> {result.suggestion}</p>
                <p><strong>Top Words:</strong> {result.top_words.slice(0, 8).map(w => w.word).join(', ')}</p>
              </>
            )}
          </div>
        )}

        {/* Trend Section */}
        <div className="section" style={{ textAlign: 'center' }}>
          <button className="btn" onClick={() => setShowTrend(v => !v)}>
            {showTrend ? 'Hide Trend' : 'Show Trend'}
          </button>
          {showTrend && trendData.length > 0 && (
            <>
              <div style={{ margin: '16px 0' }}>
                <span style={{ marginRight: 8 }}>Trend Type:</span>
                <select value={trendType} onChange={e => setTrendType(e.target.value)} style={{ padding: '4px 8px', fontSize: 14 }}>
                  <option value="compound">Fusion</option>
                  <option value="compound_vader">VADER</option>
                  <option value="compound_bert">BERT</option>
                </select>
              </div>
              <div style={{ marginTop: 10, height: 300 }}>
                <ResponsiveContainer>
                  <LineChart data={trendData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#000' }} />
                    <YAxis domain={[-1, 1]} tick={{ fill: '#000' }} />
                    <Tooltip />
                    <Line type="monotone" dataKey={trendType} stroke="#000" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>

        {/* Word Cloud Section */}
        <div className="section" style={{ textAlign: 'center' }}>
          <button className="btn" onClick={() => setShowWordCloud(v => !v)}>
            {showWordCloud ? 'Hide Word Cloud' : 'Show Word Cloud'}
          </button>
          {showWordCloud && wordCloudData.length > 0 && (
            <div style={{ marginTop: 20, height: 360 }}>
              <ReactECharts
                option={{
                  series: [{
                    type: 'wordCloud',
                    left: 'center',
                    top: 'center',
                    width: '100%',
                    height: '100%',
                    textStyle: { color: '#000' },
                    data: wordCloudData
                  }]
                }}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
