from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import os
import time
import uuid
import cv2
import torch


logger = logging.getLogger(__name__)

# Set PREDICTOR_DEBUG=1 in the environment to enable verbose per-prediction
# debug logging (model load, class distribution, timing, confidence, etc.)
DEBUG = os.environ.get("PREDICTOR_DEBUG", "").lower() in ("1", "true", "yes")


# =====================================================
# ERROR CLASSIFICATION
# =====================================================

class PredictorError(Exception):
    """Base class for all predictor errors."""


class ModelLoadError(PredictorError):
    """Raised when the YOLO model fails to load from disk."""


class ImageError(PredictorError):
    """Raised when an input frame/image is missing, empty, or malformed."""


class InferenceError(PredictorError):
    """Raised when the underlying YOLO model call fails."""


class ParsingError(PredictorError):
    """Raised when YOLO results cannot be parsed into detections."""


class NormalizationError(PredictorError):
    """Raised when a label cannot be normalized."""


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
    MODEL_VERSION = "1.0.0"

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

        self.model_loaded = False

        try:
            self.model = YOLO(
                str(self.model_path)
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load {self.MODEL_SOURCE} model "
                f"from {self.model_path}: {exc}"
            ) from exc

        if self.device == "cuda":
            self.model.to("cuda")
            self.model.fuse()

        self.model_name = model_name
        self.model_loaded = True

        # Running counters across the lifetime of this predictor instance.
        self.prediction_count = 0

        logger.info(
            f"{self.MODEL_SOURCE} model loaded successfully"
        )

        if DEBUG:
            logger.debug(
                f"[DEBUG] Loaded model={self.model_name} "
                f"source={self.MODEL_SOURCE} device={self.device} "
                f"version={self.MODEL_VERSION}"
            )

    # -------------------------------------------------

    def predict_frame(
        self,
        frame
    ) -> PredictionResult:

        total_start = time.time()

        prediction_status = "SUCCESS"
        error_type = None
        error_message = None

        detections: List[Dict[str, Any]] = []
        raw_box_count = 0
        rejected_count = 0

        preprocess_time = 0.0
        inference_time = 0.0
        parsing_time = 0.0

        image_height = image_width = image_channels = None

        try:
            # ---- Input validation -------------------------------------
            if frame is None:
                raise ImageError("Input frame is None")

            if hasattr(frame, "size") and frame.size == 0:
                raise ImageError("Input frame is empty")

            if hasattr(frame, "shape"):
                shape = frame.shape
                image_height = int(shape[0]) if len(shape) > 0 else None
                image_width = int(shape[1]) if len(shape) > 1 else None
                image_channels = int(shape[2]) if len(shape) > 2 else 1

            # ---- Preprocess ---------------------------------------------
            t0 = time.time()
            try:
                frame = self.preprocess(frame)
            except Exception as exc:
                raise ImageError(f"Preprocessing failed: {exc}") from exc
            preprocess_time = round(time.time() - t0, 4)

            # ---- Inference ------------------------------------------------
            t0 = time.time()
            try:
                results = self.model(
                    frame,
                    device=self.device,
                    conf=self.config.confidence,
                    iou=self.config.iou,
                    imgsz=self.config.imgsz,
                    verbose=False
                )
            except Exception as exc:
                raise InferenceError(f"Model inference failed: {exc}") from exc
            inference_time = round(time.time() - t0, 4)

            # ---- Validate results ------------------------------------------
            self._validate_results(results)

            # ---- Parse -----------------------------------------------------
            t0 = time.time()
            try:
                detections, raw_box_count, rejected_count = self._parse_results_tracked(
                    results
                )
            except Exception as exc:
                raise ParsingError(f"Failed to parse results: {exc}") from exc
            parsing_time = round(time.time() - t0, 4)

        except PredictorError as exc:
            prediction_status = "FAILED"
            error_type = type(exc).__name__
            error_message = str(exc)
            logger.error(f"{self.MODEL_SOURCE} prediction failed: {error_type}: {error_message}")

        self.prediction_count += 1

        total_time = round(time.time() - total_start, 4)

        class_distribution = self._compute_class_distribution(detections)
        confidence_stats = self._compute_confidence_stats(detections)

        metadata: Dict[str, Any] = {
            "model_name": self.model_name,
            "model_source": self.MODEL_SOURCE,
            "model_version": self.MODEL_VERSION,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "prediction_status": prediction_status,
            "prediction_count": self.prediction_count,

            "timing": {
                "preprocessing": preprocess_time,
                "inference": inference_time,
                "parsing": parsing_time,
                "total": total_time,
            },
            # Kept for backwards compatibility with existing consumers.
            "inference_time": total_time,

            "detection_counters": {
                "raw_boxes": raw_box_count,
                "rejected_boxes": rejected_count,
                "returned_boxes": len(detections),
            },

            "image_metadata": {
                "width": image_width,
                "height": image_height,
                "channels": image_channels,
            },

            "class_distribution": class_distribution,
            "confidence_stats": confidence_stats,

            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if error_type:
            metadata["error_type"] = error_type
            metadata["error_message"] = error_message

        if DEBUG:
            self._log_debug_summary(detections, metadata)

        return PredictionResult(
            detections=detections,
            metadata=metadata,
        )

    # -------------------------------------------------

    def predict_batch(
        self,
        frames
    ):

        batch_start = time.time()

        frames = [
            self.preprocess(f)
            for f in frames
        ]

        try:
            results = self.model(
                frames,
                device=self.device,
                conf=self.config.confidence,
                iou=self.config.iou,
                imgsz=self.config.imgsz,
                verbose=False
            )
        except Exception as exc:
            raise InferenceError(f"Batch inference failed: {exc}") from exc

        per_image_detections = [
            self.parse_single(r)
            for r in results
        ]

        total_time = round(time.time() - batch_start, 4)
        total_objects = sum(len(d) for d in per_image_detections)
        batch_size = len(frames)

        batch_metadata = {
            "model_name": self.model_name,
            "model_source": self.MODEL_SOURCE,
            "batch_size": batch_size,
            "images_processed": batch_size,
            "total_objects": total_objects,
            "total_time": total_time,
            "average_time_per_image": (
                round(total_time / batch_size, 4) if batch_size else 0.0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if DEBUG:
            logger.debug(f"[DEBUG] Batch summary: {batch_metadata}")

        return per_image_detections, batch_metadata

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

        detections, _, _ = self._parse_single_tracked(result)
        return detections

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _validate_results(self, results):
        """Raise ParsingError if YOLO's return value looks unusable."""

        if results is None:
            raise ParsingError("Model returned None instead of results")

        if not isinstance(results, (list, tuple)) and not hasattr(results, "__iter__"):
            raise ParsingError(f"Model returned non-iterable results: {type(results)}")

        results_list = list(results)

        if len(results_list) == 0:
            raise ParsingError("Model returned an empty results list")

        for result in results_list:
            if not hasattr(result, "boxes"):
                raise ParsingError("Result object missing 'boxes' attribute")
            if not hasattr(result, "names"):
                raise ParsingError("Result object missing 'names' attribute")

    def _parse_results_tracked(self, results):
        """Like parse_results, but also returns raw/rejected box counts."""

        all_detections = []
        total_raw = 0
        total_rejected = 0

        for result in results:
            detections, raw_count, rejected_count = self._parse_single_tracked(result)
            all_detections.extend(detections)
            total_raw += raw_count
            total_rejected += rejected_count

        return all_detections, total_raw, total_rejected

    def _parse_single_tracked(self, result):
        """Parses one YOLO result, validating each box.

        Returns (detections, raw_box_count, rejected_count).
        """

        detections = []

        if result.boxes is None:
            return detections, 0, 0

        raw_box_count = len(result.boxes)
        rejected_count = 0

        for box in result.boxes:

            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
            except Exception as exc:
                rejected_count += 1
                logger.warning(f"Skipping malformed box: {exc}")
                continue

            # ---- Per-detection validation --------------------------------
            if x2 <= x1 or y2 <= y1:
                rejected_count += 1
                logger.warning(
                    f"Skipping detection with invalid bbox geometry: "
                    f"({x1}, {y1}, {x2}, {y2})"
                )
                continue

            if not (0.0 <= conf <= 1.0):
                rejected_count += 1
                logger.warning(f"Skipping detection with out-of-range confidence: {conf}")
                continue

            if cls not in result.names:
                rejected_count += 1
                logger.warning(f"Skipping detection with unknown class id: {cls}")
                continue

            raw_label = result.names[cls]

            try:
                label = self.normalize_label(raw_label)
            except Exception as exc:
                raise NormalizationError(
                    f"Failed to normalize label '{raw_label}': {exc}"
                ) from exc

            detections.append({

                "detection_id": str(uuid.uuid4()),

                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ],

                "class_id": cls,

                "class_name": label,
                "raw_class_name": str(raw_label),

                "confidence":
                    round(conf, 3),

                "model_source":
                    self.MODEL_SOURCE,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat()

            })

        return detections, raw_box_count, rejected_count

    def _compute_class_distribution(self, detections):
        distribution: Dict[str, int] = {}
        for det in detections:
            name = det["class_name"]
            distribution[name] = distribution.get(name, 0) + 1
        return distribution

    def _compute_confidence_stats(self, detections):
        if not detections:
            return {
                "highest": None,
                "lowest": None,
                "average": None,
            }

        confidences = [d["confidence"] for d in detections]
        return {
            "highest": round(max(confidences), 3),
            "lowest": round(min(confidences), 3),
            "average": round(sum(confidences) / len(confidences), 3),
        }

    def _log_debug_summary(self, detections, metadata):
        lines = [
            "============================",
            "Prediction Summary",
            "============================",
            f"Model          : {self.model_name}",
            f"Status         : {metadata['prediction_status']}",
            f"Raw Boxes      : {metadata['detection_counters']['raw_boxes']}",
            f"Rejected       : {metadata['detection_counters']['rejected_boxes']}",
            f"Returned       : {metadata['detection_counters']['returned_boxes']}",
            "",
            "Classes",
        ]
        for cls_name, count in metadata["class_distribution"].items():
            lines.append(f"  {cls_name:<14} : {count}")

        stats = metadata["confidence_stats"]
        lines += [
            "",
            "Confidence",
            f"  Highest      : {stats['highest']}",
            f"  Average      : {stats['average']}",
            f"  Lowest       : {stats['lowest']}",
            "",
            f"Inference Time : {metadata['timing']['inference'] * 1000:.1f} ms",
            "============================",
        ]
        logger.debug("[DEBUG] " + "\n".join(lines))

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

        resolved = LABEL_MAP.get(
            label,
            LABEL_MAP.get(
                normalized,
                normalized
            )
        )

        if DEBUG and resolved != normalized:
            logger.debug(f"[DEBUG] Normalized label: '{label}' -> '{resolved}'")

        return resolved


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