import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import StockBasic from './pages/StockBasic';
import StockDetail from './pages/StockDetail';
import Portfolio from './pages/Portfolio';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/stock" element={<StockBasic />} />
        <Route path="/portfolio" element={<Portfolio />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
