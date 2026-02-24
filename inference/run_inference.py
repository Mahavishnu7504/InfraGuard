import json
from pathlib import Path
from src.inference.predictor import YOLOPredictor
from src.data_pipeline.ppe_violation import detect_ppe_violations

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/infra_ppe_cpu_safe/weights/best.pt"
SOURCE_DIR = "inference/test_images"
OUTPUT_DIR = "inference/outputs"
RUN_NAME = "ppe_results"
JSON_DIR = Path("inference/outputs/json")
# ----------------------------------------

print("🚀 InfraGuard inference started")
print(f"📂 Reading images from: {SOURCE_DIR}")

# Ensure output directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Load predictor
predictor = YOLOPredictor(
    model_path=MODEL_PATH,
    conf=0.25,
    imgsz=416
)

# Run YOLO
results = predictor.predict(
    source=SOURCE_DIR,
    save=True,
    project=OUTPUT_DIR,
    name=RUN_NAME
)

print(f"🧠 YOLO returned {len(results)} result objects")

# Post-processing
for r in results:
    detections = []
    names = r.names

    if r.boxes is None:
        print("⚠️ No boxes detected in image:", r.path)
        continue

    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": names[cls_id],
            "box": box.xyxy[0].tolist()
        })

    # 🔹 PPE violation analysis
    violation_result = detect_ppe_violations(detections)

    image_name = Path(r.path).name

    output_json = {
        "image": image_name,
        "risk": violation_result["risk"],
        "violations": violation_result["violations"],
        "detections": detections
    }

    json_path = JSON_DIR / f"{image_name}.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # Console output
    print(f"\n📷 Image: {image_name}")
    print(f"🚨 Risk Level: {violation_result['risk']}")
    for v in violation_result["violations"]:
        print("⚠️", v)

    print(f"🧾 JSON saved: {json_path}")

print("✅ Inference completed")