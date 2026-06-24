from ultralytics import YOLO

model = YOLO(r"F:\InfraGuard\models\crack.pt")

results = model.predict(
    source=r"F:\InfraGuard\tests\images\wall.jpg",
    conf=0.25,
    save=True
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        print(
            f"Class: {model.names[cls]} | Confidence: {conf:.2f}"
        )

print("Crack Detection Complete")