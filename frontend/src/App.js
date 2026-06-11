import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './auth/AuthContext';
import ProtectedRoute from './auth/ProtectedRoute';
import { PERMISSIONS } from './auth/permissions';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';

import Landing from './pages/Landing';
import Login from './pages/Login';
import ExecutiveSummary from './pages/ExecutiveSummary';
import CommodityIntelligence from './pages/CommodityIntelligence';
import FinancialPnL from './pages/FinancialPnL';
import ScenarioSimulation from './pages/ScenarioSimulation';
import MarketMonitor from './pages/MarketMonitor';
import Governance from './pages/Governance';
import DataExplorer from './pages/DataExplorer';
import InsightsCenter from './pages/InsightsCenter';
import VarianceBridge from './pages/VarianceBridge';
import WarrantyAnalytics from './pages/WarrantyAnalytics';

function AppShell() {
  return (
    <div className="flex h-screen overflow-hidden" style={{ backgroundColor: '#0f172a' }}>
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main
          className="flex-1 overflow-y-auto p-6 scrollbar-thin"
          style={{ backgroundColor: '#0f172a' }}
        >
          <Routes>
            <Route index element={<Navigate to="executive" replace />} />
            <Route path="executive" element={<ExecutiveSummary />} />
            <Route path="commodity" element={<CommodityIntelligence />} />
            <Route path="pnl" element={<FinancialPnL />} />
            <Route path="simulation" element={<ScenarioSimulation />} />
            <Route path="market" element={<MarketMonitor />} />
            <Route path="insights" element={<InsightsCenter />} />
            <Route path="variance" element={<VarianceBridge />} />
            <Route path="warranty" element={<WarrantyAnalytics />} />
            <Route path="governance" element={<Governance />} />
            <Route
              path="data"
              element={
                <ProtectedRoute permission={PERMISSIONS.VIEW_AGGREGATED_DATA}>
                  <DataExplorer />
                </ProtectedRoute>
              }
            />
            <Route path="*" element={<Navigate to="executive" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/app/*"
            element={
              <ProtectedRoute>
                <AppShell />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
