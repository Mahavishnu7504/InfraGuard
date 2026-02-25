import cv2
from pathlib import Path
from src.inference.predictor import YOLOPredictor
from src.data_pipeline.ppe_violation import detect_ppe_violations

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/infra_ppe_cpu_safe/weights/best.pt"
VIDEO_SOURCE = "inference/test_video/sample.mp4"  # change later
CONF = 0.25
IMG_SIZE = 416
# ----------------------------------------

print("🎥 InfraGuard video inference started")

# Load model
predictor = YOLOPredictor(
    model_path=MODEL_PATH,
    conf=CONF,
    imgsz=IMG_SIZE
)

# Open video (file or webcam)
cap = cv2.VideoCapture(0 if VIDEO_SOURCE == "webcam" else VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video source")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # YOLO inference on frame
    results = predictor.model.predict(
        source=frame,
        conf=CONF,
        imgsz=IMG_SIZE,
        verbose=False
    )

    for r in results:
        detections = []
        names = r.names

        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": names[cls_id],
                "box": box.xyxy[0].tolist()
            })

        violations = detect_ppe_violations(detections)

        # Draw results on frame
        for v in violations:
            text = f"Person {v['person_id']} | Risk: {v['risk']}"
            cv2.putText(
                frame,
                text,
                (20, 40 + v["person_id"] * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if v["risk"] == "HIGH" else (0, 255, 0),
                2
            )

    cv2.imshow("InfraGuard - PPE Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video inference finished")