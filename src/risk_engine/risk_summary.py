import json
from pathlib import Path
from collections import Counter

JSON_DIR = Path("inference/outputs/json")
OUTPUT_FILE = Path("inference/outputs/risk_summary.json")

def generate_risk_summary():
    risk_counter = Counter()
    total_images = 0

    for json_file in JSON_DIR.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        risk = data.get("risk", "UNKNOWN")
        risk_counter[risk] += 1
        total_images += 1

    if total_images == 0:
        raise RuntimeError("No JSON files found for risk summary")

    compliance_score = (
        (risk_counter["LOW"] / total_images) * 100
        if total_images > 0 else 0
    )

    summary = {
        "total_images": total_images,
        "risk_breakdown": dict(risk_counter),
        "compliance_score_percent": round(compliance_score, 2)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("📊 Risk Summary Generated")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate_risk_summary()