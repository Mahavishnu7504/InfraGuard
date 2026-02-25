# src/risk_engine/risk_summary.py

import json
from pathlib import Path
from collections import Counter

JSON_DIR = Path("inference/outputs/json")

def generate_risk_summary():
    json_files = list(JSON_DIR.glob("*.json"))
    if not json_files:
        raise RuntimeError("No inference JSON files found")

    risks = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
            risk = data.get("final_risk", "UNKNOWN")
            risks.append(risk)

    total = len(risks)
    breakdown = Counter(risks)

    compliant = breakdown.get("LOW", 0)
    compliance_score = round((compliant / total) * 100, 2)

    summary = {
        "total_images": total,
        "risk_breakdown": dict(breakdown),
        "compliance_score_percent": compliance_score
    }

    print("📊 Risk Summary Generated")
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    generate_risk_summary()