# =========================================================
# QUALITY ASSURANCE ROUTES
# =========================================================

import uuid
import shutil
import logging
import traceback

from pathlib import Path
from datetime import datetime, timezone
from typing import List

import cv2

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Body
)

from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import Optional, Any

from backend.services.quality.analysis_service import (
    analyze_quality
)

from backend.services.quality.pdf_service import (
    generate_quality_pdf
)

# =========================================================
# LOGGER                                  (Priority 3)
# =========================================================

logger = logging.getLogger(__name__)

# =========================================================
# ROUTER
# =========================================================

router = APIRouter(
    tags=["Quality Assurance"]
)

# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = Path(
    "backend/uploads/quality"
)

REPORT_DIR = Path(
    "backend/uploads/reports"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# CONFIG
# =========================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

# Priority 1 — MIME type whitelist
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)

# Priority 6 — server-side image count cap
MAX_IMAGES = 20

# =========================================================
# PYDANTIC REQUEST MODEL                  (Priority 9)
# =========================================================

class ImagePayload(BaseModel):
    image_id:              Optional[str]  = None
    image_index:           Optional[int]  = None
    original_filename:     Optional[str]  = None
    report:                Optional[list] = None
    overall_status:        Optional[str]  = None
    compliance_score:      Optional[Any]  = None
    inspection_grade:      Optional[str]  = None
    overall_risk:          Optional[str]  = None
    executive_summary:     Optional[str]  = None
    priority_action:       Optional[str]  = None
    analytics:             Optional[dict] = None
    ai_findings:           Optional[list] = None
    image_path:            Optional[str]  = None
    annotated_image_path:  Optional[str]  = None


class QualityReportRequest(BaseModel):
    inspection_id:      Optional[str] = None
    project_name:       Optional[str] = None
    site_location:      Optional[str] = None
    inspector_name:     Optional[str] = None
    client_name:        Optional[str] = None
    project_phase:      Optional[str] = None
    contractor:         Optional[str] = None
    review_status:      Optional[str] = None
    prepared_by:        Optional[str] = None
    reviewed_by:        Optional[str] = None
    approved_by:        Optional[str] = None
    images:             List[ImagePayload] = []

# =========================================================
# VALIDATION
# =========================================================

def validate_image_file(
    file: UploadFile
):
    # Priority 1 — check extension
    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(

            status_code=400,

            detail=(
                "Only JPG, JPEG and PNG supported."
            )
        )

    # Priority 1 — check MIME type (guards against spoofed extensions)
    if file.content_type not in ALLOWED_MIME_TYPES:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Unsupported image format: {file.content_type}. "
                "Only image/jpeg and image/png are accepted."
            )
        )

# =========================================================
# MOCK AI DETECTIONS
# =========================================================

def mock_detections():

    return [

        {
            "issue_type":
                "surface_crack",

            "category":
                "Structural Integrity",

            "severity":
                "High",

            "confidence":
                0.92,

            "bbox":
                [120, 100, 360, 220]
        },

        {
            "issue_type":
                "water_leakage",

            "category":
                "Moisture Damage",

            "severity":
                "Medium",

            "confidence":
                0.81,

            "bbox":
                [420, 260, 620, 410]
        }
    ]

# =========================================================
# HELPERS
# =========================================================

def _native_path(value: str) -> str:
    """
    Converts a forward-slash-normalised path string (as stored in
    JSON API responses) back to the native OS path representation
    recognised by disk I/O operations (cv2.imread, open(), etc.).

    Returns an empty string when value is falsy so callers can use
    a simple truthiness check before opening the file.
    """
    if not value:
        return ""
    return str(Path(value))

# =========================================================
# PROCESS SINGLE IMAGE
# =========================================================

def process_single_image(
    file: UploadFile,
    image_index: int
):
    """
    Validates, saves, and analyzes one uploaded image.
    Returns a per-image result dict or raises HTTPException.
    Cleans up the saved file if any processing step fails.  (Priority 7)
    """

    validate_image_file(file)

    extension = Path(
        file.filename
    ).suffix.lower()

    image_id = uuid.uuid4().hex

    file_name = (
        f"{image_id}{extension}"
    )

    image_path = (
        UPLOAD_DIR / file_name
    )

    # Track whether the file was successfully written so the
    # finally block knows whether there is anything to remove.
    file_saved = False

    try:

        with open(image_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        file_saved = True

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Image {image_index + 1} "
                    f"({file.filename}) processing failed."
                )
            )

        detections = mock_detections()

        result = analyze_quality(
            str(image_path),
            detections
        )

        if not result:

            raise HTTPException(

                status_code=500,

                detail=(
                    f"AI analysis failed for image "
                    f"{image_index + 1} ({file.filename})."
                )
            )

    except Exception:

        # Priority 7 — remove orphan file on any failure after save
        if file_saved:
            image_path.unlink(missing_ok=True)
            logger.debug(
                "Cleaned up orphan upload: %s",
                image_path
            )

        raise

    # Normalise both paths to forward slashes for JSON transport.
    # _native_path() in download_report converts them back to OS
    # paths before passing them to generate_quality_pdf().
    raw_image_path = str(image_path).replace("\\", "/")

    raw_annotated_path = str(
        result.get("annotated_image_path") or ""
    ).replace("\\", "/")

    return {

        "image_id":
            image_id,

        "image_index":
            image_index + 1,

        "original_filename":
            file.filename,

        "report":
            result.get("report", []),

        "overall_status":
            result.get("overall_status", "Unknown"),

        "compliance_score":
            result.get("compliance_score", 0),

        "inspection_grade":
            result.get("inspection_grade", "N/A"),

        "overall_risk":
            result.get("overall_risk", "Unknown"),

        "executive_summary":
            result.get("executive_summary", ""),

        "priority_action":
            result.get("priority_action", ""),

        "analytics":
            result.get("analytics", {}),

        "ai_findings":
            result.get("ai_findings", []),

        # Forward-slash paths safe for JSON transport and frontend display
        "image_path":
            raw_image_path,

        "annotated_image_path":
            raw_annotated_path,
    }

# =========================================================
# ANALYZE QUALITY — MULTI-IMAGE
# =========================================================

@router.post("/analyze")

async def analyze_quality_route(
    files: List[UploadFile] = File(...)
):
    """
    Accepts one or more images (max MAX_IMAGES).
    Analyzes each image independently.
    Returns image-wise findings without merging results.
    """

    if not files:

        raise HTTPException(

            status_code=400,

            detail="No images provided."
        )

    # Priority 6 — enforce server-side image count limit
    if len(files) > MAX_IMAGES:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Too many images. "
                f"Maximum allowed per request is {MAX_IMAGES}."
            )
        )

    # Priority 10 — timezone-aware timestamp
    inspection_id = uuid.uuid4().hex

    image_results = []

    errors = []

    for index, file in enumerate(files):

        try:

            contents = await file.read()

            # Priority 5 — reject empty files
            if not contents:

                errors.append({
                    "image_index": index + 1,
                    "filename": file.filename,
                    "error": "Uploaded file is empty."
                })

                continue

            # Priority 2 — size check on buffered bytes
            # (avoids holding the full stream open unnecessarily)
            if len(contents) > MAX_FILE_SIZE:

                errors.append({
                    "image_index": index + 1,
                    "filename": file.filename,
                    "error": "File exceeds 10 MB limit."
                })

                continue

            await file.seek(0)

            result = process_single_image(
                file,
                index
            )

            image_results.append(result)

        except HTTPException as e:

            errors.append({
                "image_index": index + 1,
                "filename": file.filename,
                "error": e.detail
            })

        except Exception as e:

            # Priority 3 — structured logging instead of bare traceback
            logger.exception(
                "Unexpected error processing image %d (%s)",
                index + 1,
                file.filename
            )

            errors.append({
                "image_index": index + 1,
                "filename": file.filename,
                "error": str(e)
            })

    if not image_results:

        raise HTTPException(

            status_code=500,

            detail=(
                "All images failed processing. "
                "Check errors field for details."
            )
        )

    return {

        "success":
            True,

        "inspection_id":
            inspection_id,

        # Priority 10 — timezone-aware UTC timestamp
        "timestamp":
            datetime.now(timezone.utc).isoformat(),

        "processing_engine":
            "InfraGuard Enterprise AI",

        "total_images_submitted":
            len(files),

        "total_images_processed":
            len(image_results),

        "images":
            image_results,

        "errors":
            errors
    }

# =========================================================
# DOWNLOAD PDF
# =========================================================

@router.post("/report/download")

async def download_report(
    payload: QualityReportRequest = Body(...)       # Priority 9
):

    try:

        report_id = uuid.uuid4().hex

        # --------------------------------------------------
        # Normalise frontend payload shape → pdf_service schema
        #
        # Frontend sends:
        #   images[].report[]          → findings
        #   images[].overall_status    → compliance_status
        #   images[].overall_risk      → risk_level
        #   images[].image_path        → original upload path (fwd-slash)
        #   images[].annotated_image_path → HUD evidence image (fwd-slash)
        #
        # pdf_service expects native OS paths it can open directly.
        # _native_path() converts the forward-slash JSON paths back to
        # whatever separator the host OS requires (critical on Windows).
        # --------------------------------------------------

        image_reports = []

        for img in payload.images:

            img_dict = img.model_dump()

            image_reports.append({
                **img_dict,

                # Schema aliases expected by pdf_service
                "findings":          img_dict.get("report", []),
                "compliance_status": img_dict.get("overall_status", "—"),
                "risk_level":        img_dict.get("overall_risk",   "—"),

                # Re-resolve forward-slash JSON paths → native OS paths
                # so pdf_service can open the files without path errors.
                "image_path":           _native_path(
                    img_dict.get("image_path", "")
                ),
                "annotated_image_path": _native_path(
                    img_dict.get("annotated_image_path", "")
                ),
            })

        # --------------------------------------------------
        # Upgrade 2: Report versioning
        # --------------------------------------------------
        report_version = "1.0"

        # --------------------------------------------------
        # Upgrade 5: Audit timestamp (Priority 10 — tz-aware)
        # --------------------------------------------------
        audit_generated_at = datetime.now(timezone.utc).isoformat()

        payload_dict = payload.model_dump()

        report_data = {
            **payload_dict,

            # Upgrade 1: Inspection metadata
            "project_name":     payload.project_name,
            "site_location":    payload.site_location,
            "inspector_name":   payload.inspector_name,
            "client_name":      payload.client_name,
            "project_phase":    payload.project_phase,
            "contractor":       payload.contractor,
            "review_status":    payload.review_status,

            # Upgrade 2: Versioning
            "revision":         report_version,

            # Upgrade 3: Executive approval fields
            "prepared_by":      payload.prepared_by,
            "reviewed_by":      payload.reviewed_by,
            "approved_by":      payload.approved_by,

            # Upgrade 5: Audit timestamp
            "audit_generated_at": audit_generated_at,

            "image_reports": image_reports,
        }

        # --------------------------------------------------
        # Upgrade 4: Dynamic PDF name
        # Uses inspection_id when available, falls back to
        # project_name + date, then the generic report_id.
        # --------------------------------------------------
        inspection_id = payload.inspection_id
        project_name  = payload.project_name

        if inspection_id:
            pdf_name = f"InfraGuard_QA_{inspection_id}.pdf"
        elif project_name:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            safe_name = "".join(
                c if c.isalnum() else "_"
                for c in project_name
            )
            pdf_name = f"InfraGuard_{safe_name}_{date_str}.pdf"
        else:
            pdf_name = f"InfraGuard_QA_{report_id}.pdf"

        pdf_path = REPORT_DIR / pdf_name

        generate_quality_pdf(

            report_data=report_data,

            output_path=str(pdf_path)
        )

        if not pdf_path.exists():

            raise HTTPException(

                status_code=500,

                detail="PDF generation failed."
            )

        return FileResponse(

            path=str(pdf_path),

            filename=pdf_name,

            media_type="application/pdf"
        )

    except HTTPException:
        raise

    except Exception as e:

        # Priority 3 — structured logging
        logger.exception("PDF generation failed")

        raise HTTPException(

            status_code=500,

            detail=f"PDF failed: {str(e)}"
        )