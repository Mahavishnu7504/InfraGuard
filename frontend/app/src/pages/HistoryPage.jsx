import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    FaHistory, FaShieldAlt, FaClock,
    FaDatabase, FaSearch, FaFilter,
    FaExclamationTriangle, FaCheckCircle,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAlerts } from "../services/api";
import "./history.css";

function PDot() {
    return <span className="pdot" />;
}

const ITEM_CLS = { HIGH: "hp__item--high", MEDIUM: "hp__item--medium", LOW: "hp__item--low" };
const itemCls = (r) => ITEM_CLS[(r || "LOW").toUpperCase()] || ITEM_CLS.LOW;

// Backend sends timestamps with their own timezone info already included
// (e.g. "Wed, 24 Jun 2026 06:34:22 GMT" or a full ISO string with offset/Z).
// Date() parses both correctly on its own — don't append "Z" manually,
// since that would corrupt formats that already specify a timezone.
function formatDate(timestamp) {
    const ts = new Date(timestamp);
    return ts.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}

function formatTime(timestamp) {
    const ts = new Date(timestamp);
    return ts.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const TODAY = new Date().toDateString();
const REFRESH_INTERVAL = 30000;

export default function HistoryPage() {
    const [search, setSearch] = useState("");
    const [riskFilter, setRiskFilter] = useState("ALL");

    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastRefresh, setLastRefresh] = useState(null);

    useEffect(() => {
        let cancelled = false;

        async function loadHistory() {
            try {
                const data = await getAlerts();
                if (!cancelled) {
                    setIncidents(Array.isArray(data) ? data : []);
                    setError(null);
                    setLastRefresh(new Date());
                }
            } catch (err) {
                console.error("[HISTORY ERROR]", err);
                if (!cancelled) setError("Unable to load incident history. Check your connection and try again.");
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        loadHistory();
        const iv = setInterval(loadHistory, REFRESH_INTERVAL);

        return () => {
            cancelled = true;
            clearInterval(iv);
        };
    }, []);

    // Single pass — derive all counts at once to avoid repeated filters
    const counts = useMemo(() => {
        let high = 0, medium = 0, low = 0, today = 0;
        for (const x of incidents) {
            if (x.risk_level === "HIGH") high++;
            else if (x.risk_level === "MEDIUM") medium++;
            else low++;
            if (new Date(x.timestamp).toDateString() === TODAY) today++;
        }
        return { high, medium, low, today, total: incidents.length };
    }, [incidents]);

    const filteredData = useMemo(() => {
        const query = search.trim().toLowerCase();
        return incidents.filter(item => {
            const matchesSearch =
                item.event_type?.toLowerCase().includes(query) ||
                item.description?.toLowerCase().includes(query) ||
                String(item.camera_id).includes(query);
            const matchesRisk = riskFilter === "ALL" ? true : item.risk_level === riskFilter;
            return matchesSearch && matchesRisk;
        });
    }, [search, riskFilter, incidents]);

    const sortedData = useMemo(() => {
        return [...filteredData].sort(
            (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
        );
    }, [filteredData]);

    return (
        <PageLayout>
            <div className="hp">

                {/* HEADER */}
                <motion.div className="hp__hdr" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
                    <div>
                        <div className="hp__eyebrow"><PDot /> FORENSIC TIMELINE</div>
                        <h1 className="hp__h1"><FaHistory /> Safety Incident<span className="hp__accent"> History</span></h1>
                        <p className="hp__sub">Enterprise forensic event timeline</p>
                    </div>
                    <div className="hp__hdr-right">
                        <div className="hp__status"><FaShieldAlt /> AI VERIFIED</div>
                        {lastRefresh && (
                            <div className="hp__refresh-note">
                                <PDot /> Auto-refresh every {REFRESH_INTERVAL / 1000}s · Last: {formatTime(lastRefresh)}
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* KPI — primary stats */}
                <div className="hp__kpis">
                    {[
                        { icon: <FaDatabase />, label: "Total Records", val: counts.total },
                        { icon: <FaShieldAlt />, label: "High Risk Events", val: counts.high },
                        { icon: <FaClock />, label: "Today's Events", val: counts.today },
                    ].map((k, i) => (
                        <motion.div key={k.label} className="hp__kpi"
                            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.07 }}
                        >
                            <div className="hp__kpi-icon">{k.icon}</div>
                            <div>
                                <div className="hp__kpi-val">{k.val}</div>
                                <div className="hp__kpi-label">{k.label}</div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* RISK SUMMARY — colour-coded */}
                <div className="hp__kpis">
                    {[
                        { icon: <FaExclamationTriangle />, label: "High", val: counts.high, mod: "hp__kpi--high" },
                        { icon: <FaShieldAlt />, label: "Medium", val: counts.medium, mod: "hp__kpi--medium" },
                        { icon: <FaCheckCircle />, label: "Low", val: counts.low, mod: "hp__kpi--low" },
                    ].map((k, i) => (
                        <motion.div key={k.label} className={`hp__kpi ${k.mod}`}
                            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.21 + i * 0.07 }}
                        >
                            <div className="hp__kpi-icon">{k.icon}</div>
                            <div>
                                <div className="hp__kpi-val">{k.val}</div>
                                <div className="hp__kpi-label">{k.label}</div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* CONTROLS */}
                <div className="hp__controls">
                    <div className="hp__search">
                        <FaSearch />
                        <input
                            type="text" placeholder="Search by event type, description, or camera ID..."
                            value={search} onChange={e => setSearch(e.target.value)}
                        />
                    </div>
                    <div className="hp__filter">
                        <FaFilter />
                        <select value={riskFilter} onChange={e => setRiskFilter(e.target.value)}>
                            <option>ALL</option>
                            <option>HIGH</option>
                            <option>MEDIUM</option>
                            <option>LOW</option>
                        </select>
                    </div>
                </div>

                {/* TIMELINE */}
                <div className="hp__timeline">
                    {loading
                        ? <div className="hp__empty">Loading incident history…</div>
                        : error
                            ? <div className="hp__empty hp__empty--error">{error}</div>
                            : sortedData.length > 0
                                ? sortedData.map((item, i) => (
                                    <motion.div
                                        key={item.id ?? `${item.timestamp}-${item.event_type}`}
                                        className={`hp__item ${itemCls(item.risk_level)}`}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                    >
                                        <div className="hp__item-dot" />
                                        <div className="hp__item-main">
                                            <h3>{item.event_type}</h3>
                                            <p>{item.description || "No description available"}</p>
                                        </div>
                                        <div className="hp__item-side">
                                            <span className="hp__risk">{item.risk_level}</span>
                                            <small className="hp__item-cam">CAM-{item.camera_id}</small>
                                            <small>{formatDate(item.timestamp)}</small>
                                            <small>{formatTime(item.timestamp)}</small>
                                        </div>
                                    </motion.div>
                                ))
                                : <div className="hp__empty">No incidents match your search. Try adjusting the filters.</div>
                    }
                </div>

            </div>
        </PageLayout>
    );
}