import { FaCircle } from "react-icons/fa";
import "./topbar.css";

export default function Topbar() {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <h2>InfraGuard AI Platform</h2>
        <p>CONSTRUCTION SAFETY · QUALITY INTELLIGENCE</p>
      </div>

      <div className="topbar-right">
        <div className="topbar-status">
          <FaCircle />
          <span>SYSTEM ONLINE</span>
        </div>
      </div>
    </header>
  );
}