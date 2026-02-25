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
# ---------------------------------------

print("🚀 InfraGuard inference started")

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO predictor
predictor = YOLOPredictor(
    model_path=MODEL_PATH,
    conf=0.25,
    imgsz=416
)

# Run YOLO inference
results = predictor.predict(
    source=SOURCE_DIR,
    save=True,
    project=str(OUTPUT_DIR),
    name=RUN_NAME
)

# Process each image
for r in results:
    detections = []
    names = r.names

    # Collect detections
    if r.boxes is not None:
        for box in r.boxes:
            detections.append({
                "class": names[int(box.cls[0])],
                "box": box.xyxy[0].tolist()
            })

    # PPE violation analysis (IoU-based, per person)
    analysis = detect_ppe_violations(detections)

    image_name = Path(r.path).name

    # Final structured output
    output = {
        "image": image_name,
        "image_risk": analysis.get("image_risk", "UNKNOWN"),
        "persons": analysis.get("persons", []),
        "reason": analysis.get("reason", "")
    }

    # Save JSON
    json_path = JSON_DIR / f"{image_name}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Console output
    print(f"\n📷 Image: {image_name}")
    print(f"🟥 Image Risk: {output['image_risk']}")
    for p in output["persons"]:
        print(
            f"  Person {p['person_id']} → {p['risk']} | {p['reason']}"
        )

print("\n✅ Inference completed")