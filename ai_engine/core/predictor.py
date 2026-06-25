

from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timezone
import logging
import time
import cv2
import torch


logger = logging.getLogger(__name__)


# =====================================================
# LABEL NORMALIZATION
# =====================================================

LABEL_MAP = {
    "Helmet": "helmet",
    "helmet": "helmet",

    "Vest": "vest",
    "vest": "vest",
    "vests": "vest",

    "Glove": "gloves",
    "glove": "gloves",
    "gloves": "gloves",

    "Boot": "boots",
    "boots": "boots",

    "crack": "crack",
    "crack_detection": "crack",
    "crack detection": "crack",

    "Dump Truck": "dump_truck",
    "dump truck": "dump_truck",

    "Mobile Crane": "mobile_crane",
    "mobile crane": "mobile_crane",
}


# =====================================================
# CONFIG / RESULT
# =====================================================

@dataclass
class InferenceConfig:
    confidence: float = 0.4
    iou: float = 0.5
    imgsz: int = 640
    device: str = ""


@dataclass
class PredictionResult:
    detections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# =====================================================
# BASE PREDICTOR
# =====================================================

class BasePredictor:

    MODEL_SOURCE = "unknown"

    def __init__(
        self,
        model_name: str,
        config: InferenceConfig | None = None
    ):

        self.config = (
            config
            or InferenceConfig()
        )

        root = Path(__file__).resolve().parents[2]

        self.model_path = (
            root /
            "models" /
            model_name
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )


        self.device = (
            self.config.device
            if self.config.device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )


        logger.info(
            f"Loading {self.MODEL_SOURCE} model on {self.device}"
        )


        self.model = YOLO(
            str(self.model_path)
        )


        if self.device == "cuda":
            self.model.to("cuda")
            self.model.fuse()


        self.model_name = model_name


        logger.info(
            f"{self.MODEL_SOURCE} model loaded successfully"
        )


    # -------------------------------------------------

    def predict_frame(
        self,
        frame
    ) -> PredictionResult:

        start = time.time()

        frame = self.preprocess(
            frame
        )


        results = self.model(
            frame,
            device=self.device,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            verbose=False
        )


        return PredictionResult(

            detections=
                self.parse_results(results),

            metadata={
                "model_name": self.model_name,
                "model_source": self.MODEL_SOURCE,
                "device": self.device,
                "inference_time":
                    round(
                        time.time()-start,
                        4
                    ),
                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat()
            }
        )


    # -------------------------------------------------

    def predict_batch(
        self,
        frames
    ):

        frames = [
            self.preprocess(f)
            for f in frames
        ]

        results = self.model(
            frames,
            device=self.device,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            verbose=False
        )

        return [
            self.parse_single(r)
            for r in results
        ]


    # -------------------------------------------------

    def preprocess(
        self,
        frame
    ):

        return cv2.resize(
            frame,
            (self.config.imgsz,
             self.config.imgsz)
        )


    # -------------------------------------------------

    def parse_results(
        self,
        results
    ):

        output = []

        for result in results:
            output.extend(
                self.parse_single(result)
            )

        return output


    # -------------------------------------------------

    def parse_single(
        self,
        result
    ):

        detections = []

        if result.boxes is None:
            return detections


        for box in result.boxes:

            x1,y1,x2,y2 = (
                box.xyxy[0]
                .tolist()
            )

            cls = int(
                box.cls[0]
            )

            conf = float(
                box.conf[0]
            )

            label = self.normalize_label(
                result.names[cls]
            )


            detections.append({

                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ],

                "class_id": cls,

                "class_name": label,

                "confidence":
                    round(conf, 3),

                "model_source":
                    self.MODEL_SOURCE,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat()

            })


        return detections


    # -------------------------------------------------

    def normalize_label(
        self,
        label
    ):

        normalized = (
            str(label)
            .lower()
            .replace(" ", "_")
            .strip()
        )

        return LABEL_MAP.get(
            label,
            LABEL_MAP.get(
                normalized,
                normalized
            )
        )


# =====================================================
# INFRAGUARD
# =====================================================

class InfraGuardPredictor(BasePredictor):

    MODEL_SOURCE = "infraguard"


    def __init__(self, config=None):

        super().__init__(
            "infraguard.pt",
            config
        )


# =====================================================
# CRACK
# =====================================================

class CrackPredictor(BasePredictor):

    MODEL_SOURCE = "crack"


    def __init__(self, config=None):

        super().__init__(
            "crack.pt",
            config
            or InferenceConfig(
                confidence=0.30
            )
        )
