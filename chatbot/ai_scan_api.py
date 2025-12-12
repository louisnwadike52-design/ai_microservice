"""
AI Scan API Endpoints for Django Ninja
Handles image processing and data extraction for scan-to-pay feature
"""

from ninja import Schema, Router
from typing import Dict, Any, Optional
import base64
import logging
from datetime import datetime

# Import the AI scan service
from .ai_scan_service import get_ai_scan_service

logger = logging.getLogger(__name__)

# Create router for AI scan endpoints
ai_scan_router = Router()

# --- Schemas ---

class ProcessImageRequest(Schema):
    image_data: str  # Base64 encoded image
    scan_type: str   # INVOICE, UTILITY_BILL, QR_CODE, etc.
    session_id: Optional[str] = None

class ProcessImageResponse(Schema):
    success: bool
    extracted_data: Optional[Dict[str, Any]] = None
    ai_message: Optional[str] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

class HealthResponse(Schema):
    status: str
    service: str
    version: str

class SupportedTypesResponse(Schema):
    supported_types: list

# Get AI scan service instance
scan_service = get_ai_scan_service()

# --- Endpoints ---

@ai_scan_router.get("/health", response=HealthResponse)
def health_check(request):
    """Health check endpoint for AI scan service"""
    return {
        "status": "healthy",
        "service": "ai-scan-service",
        "version": "1.0.0"
    }

@ai_scan_router.get("/supported-types", response=SupportedTypesResponse)
def get_supported_types(request):
    """Get list of supported scan types"""
    return {
        "supported_types": scan_service.supported_types
    }

@ai_scan_router.post("/process-image", response=ProcessImageResponse)
def process_image(request, payload: ProcessImageRequest):
    """
    Process an image and extract data using AI/OCR

    Request:
        - image_data: Base64 encoded image
        - scan_type: Type of document (INVOICE, UTILITY_BILL, QR_CODE, etc.)
        - session_id: Optional session identifier

    Response:
        - success: Boolean indicating if processing succeeded
        - extracted_data: Dictionary of extracted fields
        - ai_message: Helpful message from AI about the extracted data
        - confidence_score: Confidence score of the extraction (0-1)
        - error_message: Error message if processing failed
    """
    try:
        logger.info(f"Processing image for scan type: {payload.scan_type}")

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(payload.image_data)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return ProcessImageResponse(
                success=False,
                error_message=f"Failed to decode image: {str(e)}"
            )

        # Process the image using our AI scan service
        result = scan_service.process_image(image_bytes, payload.scan_type)

        if result.get('success'):
            logger.info(f"Successfully processed image. Extracted data: {list(result.get('extracted_data', {}).keys())}")
            return ProcessImageResponse(
                success=True,
                extracted_data=result.get('extracted_data', {}),
                ai_message=result.get('ai_message', ''),
                confidence_score=result.get('confidence_score', 0.0)
            )
        else:
            logger.error(f"Image processing failed: {result.get('error_message')}")
            return ProcessImageResponse(
                success=False,
                error_message=result.get('error_message', 'Unknown error'),
                extracted_data={}
            )

    except Exception as e:
        logger.exception(f"Unexpected error processing image: {e}")
        return ProcessImageResponse(
            success=False,
            error_message=f"Server error: {str(e)}"
        )

@ai_scan_router.post("/process-image-file")
def process_image_file(request):
    """
    Process an uploaded image file

    Accepts multipart/form-data with:
        - image: File upload
        - scan_type: String (INVOICE, UTILITY_BILL, etc.)
    """
    try:
        # Get uploaded file
        if 'image' not in request.FILES:
            return {
                "success": False,
                "error_message": "No image file provided"
            }

        image_file = request.FILES['image']
        scan_type = request.POST.get('scan_type', 'INVOICE')

        # Read image bytes
        image_bytes = image_file.read()

        # Process the image
        result = scan_service.process_image(image_bytes, scan_type)

        return result

    except Exception as e:
        logger.exception(f"Error processing uploaded file: {e}")
        return {
            "success": False,
            "error_message": f"Server error: {str(e)}"
        }
