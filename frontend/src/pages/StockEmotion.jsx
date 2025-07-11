import React, { useEffect, useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import ReactECharts from 'echarts-for-react';
import 'echarts-wordcloud';

export default function StockEmotion() {
  const [keyword, setKeyword] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trendData, setTrendData] = useState([]);
  const [wordCloudData, setWordCloudData] = useState([]);
  const [showTrend, setShowTrend] = useState(false);
  const [showWordCloud, setShowWordCloud] = useState(false);

  const handleEmotion = async () => {
    if (!keyword) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.get("/stock/emotion", {
        params: { keyword },
      });
      setResult(res.data);
      // 新增：更新趋势线和词云数据
      const news = res.data.news_items || [];
      setTrendData(
        news
          .filter((item) => item.publishedAt)
          .map((item) => ({
            time: item.publishedAt.slice(0, 16).replace("T", " "),
            compound: item.compound,
            title: item.title,
          }))
      );
      setWordCloudData(
        (res.data.top_words || []).map((w) => ({ text: w.word, value: w.count }))
      );
    } catch (e) {
      setResult({ error: e.response?.data?.detail || "查询失败" });
      setTrendData([]);
      setWordCloudData([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    // 页面初始不请求任何数据，趋势线和词云保持空白
    // 如需默认关键词可取消注释
    // axios.get("/stock/emotion?keyword=apple").then((res) => {
    //   const news = res.data.news_items || [];
    //   setTrendData(
    //     news
    //       .filter((item) => item.publishedAt)
    //       .map((item) => ({
    //         time: item.publishedAt.slice(0, 16).replace("T", " "),
    //         compound: item.compound,
    //         title: item.title,
    //       }))
    //   );
    //   setWordCloudData(
    //     (res.data.top_words || []).map((w) => ({ text: w.word, value: w.count }))
    //   );
    // });
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%)",
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: "40px 32px",
          borderRadius: "16px",
          boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
          minWidth: 360,
          textAlign: "center",
        }}
      >
        <h2 style={{ marginBottom: 24, color: "#f97316" }}>舆情情绪分析</h2>
        <input
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
          placeholder="请输入关键词（如 Tesla）"
          style={{
            padding: "10px 16px",
            border: "1px solid #fcd34d",
            borderRadius: 6,
            fontSize: 16,
            outline: "none",
            width: "80%",
            marginBottom: 20,
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleEmotion();
          }}
        />
        <br />
        <button
          onClick={handleEmotion}
          style={{
            background: "#f97316",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "10px 32px",
            fontSize: 16,
            cursor: "pointer",
            marginBottom: 20,
          }}
          disabled={loading}
        >
          {loading ? "分析中..." : "分析情绪"}
        </button>
        {result && (
          <div
            style={{
              marginTop: 24,
              fontSize: 16,
              color: result.error ? "#ef4444" : "#374151",
            }}
          >
            {result.error ? (
              result.error
            ) : (
              <div>
                <div>
                  <strong>关键词：</strong>
                  {result.keyword}
                </div>
                <div>
                  <strong>平均情绪分：</strong>
                  {result.avg_compound}
                </div>
                <div>
                  <strong>情绪等级：</strong>
                  {result.emotion_level}
                </div>
                <div>
                  <strong>建议：</strong>
                  <span style={{ color: "#f97316" }}>{result.suggestion}</span>
                </div>
                <div style={{ marginTop: 16, textAlign: "left" }}>
                  <strong>高频词：</strong>
                  <ul style={{ paddingLeft: 20 }}>
                    {result.top_words.slice(0, 10).map(({ word, count }, idx) => (
                      <li key={idx}>
                        {word}（{count}）
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}
        <h2 style={{ marginTop: 40, marginBottom: 16, color: "#f97316" }}>
          新闻情感趋势线
        </h2>
        <button
          style={{ marginBottom: 16, background: '#6366f1', color: '#fff', border: 'none', borderRadius: 6, padding: '8px 24px', fontSize: 15, cursor: 'pointer' }}
          onClick={() => setShowTrend((v) => !v)}
        >
          {showTrend ? '隐藏趋势线' : '显示趋势线'}
        </button>
        {showTrend && (
          <LineChart width={700} height={300} data={trendData}>
            <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis domain={[-1, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="compound" stroke="#8884d8" />
          </LineChart>
        )}

        <h2 style={{ marginTop: 40, marginBottom: 16, color: "#f97316" }}>
          情感词云图
        </h2>
        <button
          style={{ marginBottom: 16, background: '#10b981', color: '#fff', border: 'none', borderRadius: 6, padding: '8px 24px', fontSize: 15, cursor: 'pointer' }}
          onClick={() => setShowWordCloud((v) => !v)}
        >
          {showWordCloud ? '隐藏词云图' : '显示词云图'}
        </button>
        {showWordCloud && (
          <div style={{ width: 700, height: 350 }}>
            <ReactECharts
              option={{
                series: [
                  {
                    type: 'wordCloud',
                    shape: 'circle',
                    left: 'center',
                    top: 'center',
                    width: '100%',
                    height: '100%',
                    textStyle: { fontFamily: 'sans-serif' },
                    data: wordCloudData.map(w => ({ name: w.text, value: w.value })),
                  },
                ],
              }}
              style={{ width: 700, height: 350 }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
