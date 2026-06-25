import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  FaShieldAlt, FaVideo, FaClipboardCheck, FaHome, FaSearch,
} from "react-icons/fa";

import "./sidebar.css";

const MENU_ITEMS = [
  { label: "Dashboard", icon: <FaHome />, path: "/dashboard" },
  { label: "Live Monitoring", icon: <FaVideo />, path: "/safety/live" },
  { label: "Safety Detection", icon: <FaSearch />, path: "/safety/detection" },
  { label: "Quality Inspection", icon: <FaClipboardCheck />, path: "/quality/upload" },
];

export default function Sidebar() {

  return (
    <motion.aside
      className="sidebar"
      aria-label="Application Sidebar"
      initial={{ x: -40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>

        <div className="sidebar-brand">
          <div className="brand-icon"><FaShieldAlt /></div>
          <div>
            <h2>InfraGuard</h2>
            <span>Enterprise AI Platform</span>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          {MENU_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => isActive ? "sidebar-link active" : "sidebar-link"}
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span>{item.label}</span>
              <div className="sidebar-hover-glow"></div>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-dot"></span>
          AI SYSTEM ONLINE
        </div>
      </div>
    </motion.aside>
  );
}