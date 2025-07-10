import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const menuItems = [
    { label: 'Basic Info',       path: '/stock' },
    { label: 'Portfolio Viz',    path: '/portfolio' },
    { label: 'AI Score',         path: '/score' },
    { label: 'Sentiment',        path: '/emotion' },
    { label: 'Recommendations',  path: '/recommendation' }
  ];

  const [animate, setAnimate] = useState(false);
  const [pathD, setPathD]     = useState('');

  // 触发入场动画
  useEffect(() => {
    const timer = setTimeout(() => setAnimate(true), 300);
    return () => clearTimeout(timer);
  }, []);

  // 生成「金融折线」风格随机曲线路径
  useEffect(() => {
    const generateCurve = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const segments   = 200;
      const startY     = h * 0.5;
      const volatility = h * 0.08; // 波动幅度

      const points = [];
      let y = startY;
      for (let i = 0; i <= segments; i++) {
        const x = (i / segments) * w;
        y += (Math.random() - 0.5) * volatility;
        y = Math.max(0, Math.min(h, y));
        points.push([x, y]);
      }
      return points
        .map(([x, y], i) =>
          i === 0
            ? `M${x.toFixed(1)},${y.toFixed(1)}`
            : `L${x.toFixed(1)},${y.toFixed(1)}`
        )
        .join(' ');
    };

    const update = () => setPathD(generateCurve());
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return (
    <div style={{
      position: 'relative',
      minHeight: '100vh',
      background: '#fff',
      color: '#000',
      overflow: 'hidden'
    }}>
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(1000px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .fadeInUp {
          animation: fadeInUp 2.5s ease-out forwards;
        }
        @keyframes slideInRight {
          from { transform: translateX(100px); opacity: 0; }
          to   { transform: translateX(0); opacity: 1; }
        }
        .navbar {
          position: fixed;
          top: 0;
          width: 100%;
          background: #111;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          display: flex;
          justify-content: center;
          padding: 16px 0;
          z-index: 1000;
        }
        .nav-item {
          color: #aaa;
          margin: 0 24px;
          font-size: 16px;
          font-weight: 500;
          position: relative;
          padding-bottom: 6px;
          transition: color 0.3s;
          cursor: pointer;
          opacity: 0;
        }
        .nav-item:hover { color: #fff; }
        .nav-item.active { color: #fff; }
        .nav-item.active::after {
          content: '';
          position: absolute;
          bottom: 0; left: 0;
          width: 100%; height: 2px;
          background: #fff;
        }
        .footer {
          position: fixed;
          bottom: 0; width: 100%;
          background: #f9f9f9;
          border-top: 1px solid #eee;
          text-align: center;
          padding: 12px 0;
          color: #777; font-size: 14px;
          z-index: 1000;
        }
        .bg-svg {
          position: fixed;
          top: 0; left: 0;
          width: 100%; height: 100%;
          z-index: 1;
        }
      `}</style>

      {/* 背景金融折线 */}
      <svg className="bg-svg" xmlns="http://www.w3.org/2000/svg">
        <path
          d={pathD}
          stroke="#ddd"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* 导航 */}
      <nav className="navbar">
        {menuItems.map((item, idx) => {
          const isActive = location.pathname === item.path;
          return (
            <div
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`nav-item${isActive ? ' active' : ''}`}
              style={{
                animation: animate
                  ? `slideInRight 0.6s ease-out forwards ${idx * 0.15}s`
                  : 'none'
              }}
            >
              {item.label}
            </div>
          );
        })}
      </nav>

      {/* 标题 */}
      <div style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        zIndex: 2
      }}>
        <h1
          className={animate ? 'fadeInUp' : ''}
          style={{
            fontSize: '72px',
            fontFamily: '"Times New Roman", Times, serif',
            margin: 0,
            color: '#222',
            textShadow: '1px 1px 2px rgba(0,0,0,0.1)',
            letterSpacing: '2px'
          }}
        >
          FinVista
        </h1>
      </div>

      {/* 底部 */}
      <footer className="footer">
        Shen Hongli / Shao Yuxuan / Wang Lingfeng / Liu Shuyu<br />
        SUSTech / USTC / HUST
      </footer>
    </div>
  );
}
