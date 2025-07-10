import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import StockBasic from './pages/StockBasic';
import StockDetail from './pages/StockDetail';
import Portfolio from './pages/Portfolio';
import StockScore from './pages/StockScore';
import StockEmotion from './pages/StockEmotion';
import StockRecommendation from './pages/StockRecommendation';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/stock" element={<StockBasic />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/score" element={<StockScore />} />
        <Route path="/emotion" element={<StockEmotion />} />
        <Route path="/recommendation" element={<StockRecommendation />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
