"""
Upload request/response models
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class UploadRequest(BaseModel):
    """Upload request model"""
    userid: str
    devicetoken: str
    pic: str  # base64 encoded image
    id: str
    heading: Optional[float] = None
    imageMetadata: Optional[Dict[str, Any]] = None
    location: str  # JSON string
    operatingSystem: str
    barometerData: Optional[Dict[str, Any]] = None
    magnetometerData: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    """Upload response model"""
    success: bool
    truth: Optional[str] = None  # base64 encoded watermarked image
    message: Optional[str] = None
    error: Optional[str] = None
