import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import ExecutiveSummary from './pages/ExecutiveSummary';
import CommodityIntelligence from './pages/CommodityIntelligence';
import FinancialPnL from './pages/FinancialPnL';
import ScenarioSimulation from './pages/ScenarioSimulation';
import MarketMonitor from './pages/MarketMonitor';
import Governance from './pages/Governance';
import DataExplorer from './pages/DataExplorer';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden" style={{ backgroundColor: '#0f172a' }}>
        <Sidebar />
        <div className="flex flex-col flex-1 overflow-hidden">
          <Header />
          <main
            className="flex-1 overflow-y-auto p-6 scrollbar-thin"
            style={{ backgroundColor: '#0f172a' }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/executive" replace />} />
              <Route path="/executive" element={<ExecutiveSummary />} />
              <Route path="/commodity" element={<CommodityIntelligence />} />
              <Route path="/pnl" element={<FinancialPnL />} />
              <Route path="/simulation" element={<ScenarioSimulation />} />
              <Route path="/market" element={<MarketMonitor />} />
              <Route path="/governance" element={<Governance />} />
              <Route path="/data" element={<DataExplorer />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
