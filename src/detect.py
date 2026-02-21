import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "../models/yolov8n.pt"
OUTPUT_PATH = "../outputs/output.jpg"

def run_detection():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening camera")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated = results[0].plot()

        cv2.imshow("InfraGuard Detection", annotated)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()