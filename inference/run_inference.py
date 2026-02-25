import json
from pathlib import Path
from src.inference.predictor import YOLOPredictor
from src.data_pipeline.ppe_violation import detect_ppe_violations

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/infra_ppe_cpu_safe/weights/best.pt"
SOURCE_DIR = "inference/test_images"
OUTPUT_DIR = Path("inference/outputs")
JSON_DIR = OUTPUT_DIR / "json"
RUN_NAME = "ppe_results"
# ----------------------------------------

print("🚀 InfraGuard inference started")
print(f"📂 Reading images from: {SOURCE_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

predictor = YOLOPredictor(
    model_path=MODEL_PATH,
    conf=0.25,
    imgsz=416
)

results = predictor.predict(
    source=SOURCE_DIR,
    save=True,
    project=str(OUTPUT_DIR),
    name=RUN_NAME
)

print(f"🧠 YOLO returned {len(results)} result objects")

for r in results:
    detections = []
    names = r.names

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": names[cls_id],
                "box": box.xyxy[0].tolist()
            })

    violation_result = detect_ppe_violations(detections)

    image_name = Path(r.path).name

    output_json = {
        "image": image_name,
        "risk": violation_result["risk"],
        "violations": violation_result["violations"]
    }

    json_path = JSON_DIR / f"{image_name}.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"\n📷 {image_name}")
    print(f"⚠️ Risk: {output_json['risk']}")
    for v in output_json["violations"]:
        print(f"   - {v}")

print("✅ Inference completed")