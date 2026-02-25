import json
from pathlib import Path
from src.inference.predictor import YOLOPredictor
from src.data_pipeline.ppe_violation import detect_ppe_violations

MODEL_PATH = "runs/detect/infra_ppe_cpu_safe/weights/best.pt"
SOURCE_DIR = "inference/test_images"
OUTPUT_DIR = Path("inference/outputs")
JSON_DIR = OUTPUT_DIR / "json"
RUN_NAME = "ppe_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 InfraGuard inference started")

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

for r in results:
    detections = []
    names = r.names

    if r.boxes is not None:
        for box in r.boxes:
            detections.append({
                "class": names[int(box.cls[0])],
                "box": box.xyxy[0].tolist()
            })

    result = detect_ppe_violations(detections)

    image_name = Path(r.path).name

    output = {
        "image": image_name,
        "final_risk": result["risk"],
        "persons": result.get("persons", [])
    }

    with open(JSON_DIR / f"{image_name}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"{image_name} → Risk: {output['final_risk']}")

print("✅ Inference completed")