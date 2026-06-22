import {
  BrowserRouter,
  Routes,
  Route,
  Navigate
} from "react-router-dom";

/* ===================================================== */
/* CORE */
/* ===================================================== */

import Dashboard from "./pages/Dashboard";
import LiveCamera from "./pages/LiveCamera";
import InfraDetection from "./pages/InfraDetection";

/* ===================================================== */
/* QUALITY */
/* ===================================================== */

import QualityAssurance from "./pages/QualityAssurance";
import UploadImages from "./pages/quality/UploadImages";
import GenerateReport from "./pages/quality/GenerateReport";
import BestPractices from "./pages/quality/BestPractices";

/* ===================================================== */
/* ANALYTICS */
/* ===================================================== */

import AnalyticsDashboard from "./pages/AnalyticsDashboard";
import AlertCenter from "./pages/AlertCenter";
import HistoryPage from "./pages/HistoryPage";

/* ===================================================== */
/* APP */
/* ===================================================== */

export default function App() {

  return (

    <BrowserRouter>

      <Routes>

        {/* ================================================= */}
        {/* DEFAULT */}
        {/* ================================================= */}

        <Route
          path="/"
          element={<Navigate to="/dashboard" replace />}
        />

        {/* ================================================= */}
        {/* DASHBOARD */}
        {/* ================================================= */}

        <Route
          path="/dashboard"
          element={<Dashboard />}
        />

        {/* ================================================= */}
        {/* SAFETY */}
        {/* ================================================= */}

        <Route
          path="/safety/live"
          element={<LiveCamera />}
        />

        <Route
          path="/safety/detection"
          element={<InfraDetection />}
        />

        {/* ================================================= */}
        {/* QUALITY */}
        {/* ================================================= */}

        <Route
          path="/quality"
          element={<QualityAssurance />}
        />

        <Route
          path="/quality/upload"
          element={<UploadImages />}
        />

        <Route
          path="/quality/report"
          element={<GenerateReport />}
        />

        <Route
          path="/quality/best-practices"
          element={<BestPractices />}
        />

        {/* ================================================= */}
        {/* ANALYTICS */}
        {/* ================================================= */}

        <Route
          path="/analytics"
          element={<AnalyticsDashboard />}
        />

        <Route
          path="/alerts"
          element={<AlertCenter />}
        />

        <Route
          path="/history"
          element={<HistoryPage />}
        />

        {/* ================================================= */}
        {/* FALLBACK */}
        {/* ================================================= */}

        <Route
          path="*"
          element={<Navigate to="/dashboard" replace />}
        />

      </Routes>

    </BrowserRouter>
  );
}