import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import StockBasic from './pages/StockBasic';
import StockDetail from './pages/StockDetail';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<StockBasic />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
