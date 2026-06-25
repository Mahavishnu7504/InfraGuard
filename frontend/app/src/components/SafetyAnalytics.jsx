import { useMemo } from "react";
import {
  AreaChart, Area,
  BarChart, Bar,
  XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Cell,
} from "recharts";
import "./safetyanalytics.css";

/* ── Themed tooltip ──────────────────────────────────────── */
const TOOLTIP_STYLE = {
  background: "rgba(6,14,30,0.92)",
  border: "1px solid rgba(59,130,246,0.22)",
  borderRadius: 10,
  padding: "10px 14px",
  fontSize: "0.76rem",
  color: "#e2e8f0",
  backdropFilter: "blur(12px)",
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={TOOLTIP_STYLE}>
      <div style={{ color: "#93c5fd", marginBottom: 4, fontWeight: 700 }}>{label}</div>
      <div>{payload[0].value}</div>
    </div>
  );
};

const AXIS_STYLE = { fontSize: "0.7rem", fill: "#475569" };
const GRID_PROPS = { strokeDasharray: "3 3", stroke: "rgba(255,255,255,0.05)" };

/* Bar colours: low=green, medium=amber, high=red */
const RISK_COLORS = ["#22c55e", "#f59e0b", "#ef4444"];
const OPS_COLORS = ["#3b82f6", "#ef4444", "#f59e0b"];

export default function SafetyAnalytics({ data = {} }) {
  const safeData = {
    low: data.low || 0,
    medium: data.medium || 0,
    high: data.high || 0,
    compliant_workers: data.compliant_workers || 0,
    violating_workers: data.violating_workers || 0,
    danger_zones: data.danger_zones || 0,
  };

  const riskData = useMemo(() => [
    { name: "Low", value: safeData.low },
    { name: "Medium", value: safeData.medium },
    { name: "High", value: safeData.high },
  ], [safeData.low, safeData.medium, safeData.high]);

  const opsData = useMemo(() => [
    { name: "Compliant", value: safeData.compliant_workers },
    { name: "Violations", value: safeData.violating_workers },
    { name: "Danger Zones", value: safeData.danger_zones },
  ], [safeData.compliant_workers, safeData.violating_workers, safeData.danger_zones]);

  return (
    <div className="analytics-container">
      <div className="analytics-grid">

        {/* Risk Trend */}
        <div className="analytics-card">
          <div className="analytics-title">Risk Distribution</div>
          <div className="live-badge">Live</div>
          <ResponsiveContainer width="100%" height={240} aria-label="Risk distribution chart showing low, medium and high risk counts">
            <AreaChart data={riskData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid {...GRID_PROPS} />
              <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <defs>
                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#riskGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Operational Snapshot */}
        <div className="analytics-card">
          <div className="analytics-title">Operational Snapshot</div>
          <div className="live-badge">AI Summary</div>
          <ResponsiveContainer width="100%" height={240} aria-label="Operational snapshot chart showing compliant workers, violations and danger zones">
            <BarChart data={opsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid {...GRID_PROPS} />
              <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="value" radius={[8, 8, 0, 0]} maxBarSize={52}>
                {opsData.map((entry, i) => (
                  <Cell key={entry.name} fill={OPS_COLORS[i]} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
}