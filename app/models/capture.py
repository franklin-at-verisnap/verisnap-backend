"""
Capture data models
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class CaptureData(BaseModel):
    """Capture data model for database storage"""
    id: str
    userid: str
    os: str
    lat: str
    lon: str
    altitude: str
    original_signature: str
    watermarked_signature: str
    relevant_obj_tags: str
    device_time: str
    server_time: str
    t_score: str
    os_score: str
    in_vs_out_score: str
    day_vs_night_score: str
    altitude_score: str
    device_score: str
    barometerData: Optional[str] = None
    magnetometerData: Optional[str] = None
    heading: Optional[str] = None
    vscore: str
    baro_score: Optional[str] = None
    magnetometer_score: Optional[str] = None
    cheat_penalty: Optional[str] = None


class CaptureResponse(BaseModel):
    """Response model for capture data"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
