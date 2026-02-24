from pathlib import Path
from src.inference.predictor import YOLOPredictor
from src.data_pipeline.ppe_violation import detect_ppe_violations

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/infra_ppe_cpu_safe/weights/best.pt"
SOURCE_DIR = "inference/test_images"
OUTPUT_DIR = "inference/outputs"
RUN_NAME = "ppe_results"
# ----------------------------------------

print("🚀 InfraGuard inference started")
print(f"📂 Reading images from: {SOURCE_DIR}")

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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

    violations = detect_ppe_violations(detections)

    print(f"\n📷 Image: {r.path}")
    for v in violations:
        print(v)

print("✅ Inference completed")