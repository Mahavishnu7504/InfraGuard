from ultralytics import YOLO

model = YOLO(r"F:\InfraGuard\models\infraguard.pt")

for k, v in model.names.items():
    print(f"{k}: {v}")