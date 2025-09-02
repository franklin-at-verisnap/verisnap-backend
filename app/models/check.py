"""
Check request/response models
"""
from typing import Optional
from pydantic import BaseModel


class CheckRequest(BaseModel):
    """Check request model"""
    image: str  # base64 encoded image


class CheckResponse(BaseModel):
    """Check response model"""
    success: bool
    truth: Optional[str] = None  # base64 encoded verified image
    message: Optional[str] = None
    error: Optional[str] = None
