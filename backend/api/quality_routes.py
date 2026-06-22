# =========================================================
# QUALITY ASSURANCE ROUTES
# =========================================================

import uuid
import shutil
import traceback

from pathlib import Path
from datetime import datetime
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

from backend.services.quality.analysis_service import (
    analyze_quality
)

from backend.services.quality.pdf_service import (
    generate_quality_pdf
)

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

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)

# =========================================================
# VALIDATION
# =========================================================

def validate_image_file(
    file: UploadFile
):

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

    with open(image_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

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
    Accepts one or more images.
    Analyzes each image independently.
    Returns image-wise findings without merging results.
    """

    if not files:

        raise HTTPException(

            status_code=400,

            detail="No images provided."
        )

    inspection_id = uuid.uuid4().hex

    image_results = []

    errors = []

    for index, file in enumerate(files):

        try:

            contents = await file.read()

            if len(contents) > MAX_FILE_SIZE:

                errors.append({
                    "image_index": index + 1,
                    "filename": file.filename,
                    "error": "File exceeds 10MB limit."
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

            traceback.print_exc()

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

        "timestamp":
            datetime.utcnow().isoformat(),

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
    payload: dict = Body(...)
):

    try:

        report_id = uuid.uuid4().hex

        pdf_name = (
            f"report_{report_id}.pdf"
        )

        pdf_path = (
            REPORT_DIR / pdf_name
        )

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

        for img in payload.get("images", []):

            image_reports.append({
                **img,

                # Schema aliases expected by pdf_service
                "findings":          img.get("report", []),
                "compliance_status": img.get("overall_status", "—"),
                "risk_level":        img.get("overall_risk",   "—"),

                # Re-resolve forward-slash JSON paths → native OS paths
                # so pdf_service can open the files without path errors.
                "image_path":            _native_path(
                    img.get("image_path", "")
                ),
                "annotated_image_path":  _native_path(
                    img.get("annotated_image_path", "")
                ),
            })

        report_data = {
            **payload,
            "image_reports": image_reports,
        }

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

            filename=(
                "InfraGuard_Quality_Report.pdf"
            ),

            media_type="application/pdf"
        )

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=f"PDF failed: {str(e)}"
        )