import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
    FaHistory, FaShieldAlt, FaClock,
    FaDatabase, FaSearch, FaFilter,
} from "react-icons/fa";

import PageLayout from "../components/PageLayout";
import { getAlerts } from "../services/api";
import "./history.css";

function PDot() {
    return <span className="pdot" />;
}

const ITEM_CLS = { HIGH: "hp__item--high", MEDIUM: "hp__item--medium", LOW: "hp__item--low" };
const itemCls = (r) => ITEM_CLS[(r || "LOW").toUpperCase()] || ITEM_CLS.LOW;

export default function HistoryPage() {
    const [search, setSearch] = useState("");
    const [riskFilter, setRiskFilter] = useState("ALL");

    const [incidents, setIncidents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;

        async function loadHistory() {
            try {
                const data = await getAlerts();
                if (!cancelled) {
                    setIncidents(Array.isArray(data) ? data : []);
                    setError(null);
                }
            } catch (err) {
                if (!cancelled) setError("Unable to load incident history");
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        loadHistory();
        const iv = setInterval(loadHistory, 30000);

        return () => {
            cancelled = true;
            clearInterval(iv);
        };
    }, []);

    const filteredData = useMemo(() => {
        const query = search.toLowerCase();
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
                    <div className="hp__status"><FaShieldAlt /> AI VERIFIED</div>
                </motion.div>

                {/* KPI */}
                <div className="hp__kpis">
                    {[
                        { icon: <FaDatabase />, label: "Total Records", val: incidents.length },
                        {
                            icon: <FaShieldAlt />, label: "High Risk Events",
                            val: incidents.filter(x => x.risk_level === "HIGH").length
                        },
                        {
                            icon: <FaClock />, label: "Today's Events",
                            val: incidents.filter(x => {
                                const ts = new Date(x.timestamp + "Z");
                                return ts.toDateString() === new Date().toDateString();
                            }).length
                        },
                    ].map((k, i) => (
                        <motion.div key={i} className="hp__kpi"
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

                {/* RISK SUMMARY */}
                <div className="hp__kpis">
                    {[
                        { icon: <FaShieldAlt />, label: "High", val: incidents.filter(x => x.risk_level === "HIGH").length },
                        { icon: <FaShieldAlt />, label: "Medium", val: incidents.filter(x => x.risk_level === "MEDIUM").length },
                        { icon: <FaShieldAlt />, label: "Low", val: incidents.filter(x => x.risk_level === "LOW").length },
                    ].map((k, i) => (
                        <motion.div key={i} className="hp__kpi"
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
                            type="text" placeholder="Search incidents..."
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
                        ? <div className="hp__empty">Loading Incident History...</div>
                        : error
                            ? <div className="hp__empty">{error}</div>
                            : sortedData.length > 0
                                ? sortedData.map((item, i) => (
                                    <motion.div
                                        key={item.id}
                                        className={`hp__item ${itemCls(item.risk_level)}`}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                    >
                                        <div className="hp__item-dot" />
                                        <div className="hp__item-main">
                                            <h3>{item.event_type}</h3>
                                            <p>{item.description}</p>
                                        </div>
                                        <div className="hp__item-side">
                                            <span className="hp__risk">{item.risk_level}</span>
                                            <small>CAM-{item.camera_id} · {new Date(item.timestamp + "Z").toLocaleTimeString()}</small>
                                        </div>
                                    </motion.div>
                                ))
                                : <div className="hp__empty">No matching incidents found</div>
                    }
                </div>

            </div>
        </PageLayout>
    );
}