"""
Main FastAPI application for Verisnap Backend
"""
import os
import json
import gzip
import base64
import traceback
import time
from datetime import datetime, timezone
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from app.models import UploadRequest, UploadResponse, CheckRequest, CheckResponse
from app.utils import (
    ImageProcessor, 
    GeolocationService, 
    ScoringService, 
    DatabaseService, 
    AuthService
)
from PIL import Image
import imagehash

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Initialize services
image_processor = ImageProcessor()
geolocation = GeolocationService()
scoring = ScoringService()
database = DatabaseService()
auth_service = AuthService()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header"
        )
    
    user = auth_service.verify_firebase_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )
    
    return user


@app.get("/ping")
async def ping():
    """Health check endpoint"""
    return {"success": True, "message": "pong ..."}


@app.post("/check", response_model=CheckResponse)
async def check_image(request: CheckRequest, current_user: dict = Depends(get_current_user)):
    """Check if image matches any existing verified images"""
    try:
        # Decode image
        try:
            pic_data = image_processor.decode_base64_image(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        
        # Calculate hash
        try:
            with Image.open(BytesIO(pic_data)) as im:
                im = im.convert('RGB')
                avghash = imagehash.average_hash(im)
        except Exception:
            raise HTTPException(status_code=400, detail="Unreadable image")
        
        # Query database
        records = database.query_records_by_hash(str(avghash))
        if not records:
            return CheckResponse(
                success=False,
                message="Unable to verify the truth. Could not match with any existing certified picture."
            )
        
        # Check score
        rec = records[0]
        try:
            score_val = float(rec.get('vscore', '0'))
        except Exception:
            score_val = 0.0
        
        if score_val >= 75:
            final_path = os.path.join('./', f"storage/{rec['userid']}/truths/{rec['id']}.jpeg")
            try:
                with open(final_path, 'rb') as final_file:
                    photo = final_file.read()
                base64_photo = base64.b64encode(photo).decode('utf-8')
            except FileNotFoundError:
                base64_photo = None
            
            return CheckResponse(
                success=True,
                truth=base64_photo,
                message=f"Truth verified by Verisnap. TruthScore: {score_val} out of 100"
            )
        else:
            return CheckResponse(
                success=False,
                message=f"Unable to verify the truth. TruthScore of {score_val} too low."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_image(request: UploadRequest, current_user: dict = Depends(get_current_user)):
    """Upload and process image for verification"""
    try:
        # Extract location data
        location_data = json.loads(request.location)
        t = location_data.get('timestamp')
        latitude = location_data.get('latitude')
        longitude = location_data.get('longitude')
        altitude = location_data.get('altitude')
        
        # Create user directories
        userid = request.userid
        for folder in ['tmp', 'truths', 'meta', 'thumbs']:
            os.makedirs(f"storage/{userid}/{folder}", exist_ok=True)
        
        # Decode and save image
        try:
            pic_data = image_processor.decode_base64_image(request.pic)
            temp_path = f'storage/{userid}/tmp/{request.id}_temp.jpeg'
            final_path = f'storage/{userid}/truths/{request.id}.jpeg'
            meta_path = f'storage/{userid}/meta/{request.id}.json'
            
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(pic_data)
            
            with open(meta_path, 'w') as meta_file:
                meta_file.write(json.dumps(request.dict()))
                
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image and calculate scores
        current_utc_time = datetime.now(timezone.utc)
        formatted_time = current_utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        
        # Calculate various scores
        deltaT = scoring.compare_timestamp_with_current_utc(t)
        deltaOS = 5 if (scoring.compare_version_strings(request.operatingSystem, "17.0.0") == 1) else 0
        
        # Device verification
        servertimestamp = int(time.time() * 1000)
        device_result = auth_service.verify_device_token(request.devicetoken, request.id, servertimestamp)
        device_score = 10 if device_result.get('success') else -2
        
        # Image analysis
        is_p2p, r, v, ed, bd, p = image_processor.detect_picture_of_picture(temp_path)
        
        # Calculate hashes
        sha256_hash = image_processor.calculate_sha256(temp_path)
        
        # Indoor/outdoor analysis (simplified for now)
        in_vs_out = 0
        daynightscore = 0
        
        # Altitude verification
        altitudeAtLocation = geolocation.get_elevation(latitude, longitude)
        try:
            ref_alt = float(altitudeAtLocation)
            dev_alt = float(altitude)
            deltaAltitude = abs(ref_alt - dev_alt)
        except (TypeError, ValueError):
            deltaAltitude = float('inf')
        
        deltaAltitudeScore = 5 if deltaAltitude < 30.0 else -5
        
        # Magnetometer scoring
        magnetometer_score = scoring.calculate_magnetometer_score(request.dict(), request.imageMetadata)
        
        # Calculate total score
        vscore = deltaT + deltaOS + in_vs_out + daynightscore + deltaAltitudeScore + device_score + magnetometer_score
        
        # Add watermark
        image_processor.add_watermark(temp_path, final_path, f"TruthScore: {vscore}%")
        
        # Generate thumbnail
        thumb_path = f"storage/{userid}/thumbs/{request.id}_thumb.jpeg"
        image_processor.generate_thumbnail(final_path, thumb_path)
        
        # Remove temp file
        os.remove(temp_path)
        
        # Calculate final hash
        avghash = image_processor.calculate_average_hash(final_path)
        
        # Store in database
        capture_data = {
            'barometerData': json.dumps(request.barometerData) if request.barometerData else None,
            'magnetometerData': json.dumps(request.magnetometerData) if request.magnetometerData else None,
            'heading': request.heading,
            'userid': userid,
            'id': request.id,
            'os': str(request.operatingSystem),
            'lat': str(latitude),
            'lon': str(longitude),
            'altitude': str(altitude),
            'original_signature': sha256_hash,
            'watermarked_signature': avghash,
            'relevant_obj_tags': '...',
            'device_time': str(t),
            'server_time': str(formatted_time),
            't_score': str(deltaT),
            'os_score': str(deltaOS),
            'in_vs_out_score': str(in_vs_out),
            'day_vs_night_score': str(daynightscore),
            'altitude_score': str(deltaAltitudeScore),
            'device_score': str(device_score),
            'vscore': str(vscore),
            'baro_score': '0',
            'magnetometer_score': str(magnetometer_score),
            'cheat_penalty': '0'
        }
        
        database.insert_capture(capture_data)
        
        # Return watermarked image
        with open(final_path, 'rb') as final_file:
            photo = final_file.read()
        base64_photo = base64.b64encode(photo).decode('utf-8')
        
        return UploadResponse(
            success=True,
            truth=base64_photo,
            message=f'Image saved as {request.id}.jpeg with watermark'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return UploadResponse(success=False, error=str(e))


@app.get("/thumbnail/{userid}/{image_id}.jpeg")
async def serve_thumbnail(userid: str, image_id: str):
    """Serve thumbnail image"""
    directory = f"storage/{userid}/thumbs"
    filename = f"{image_id}_thumb.jpeg"
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(file_path, media_type='image/jpeg')


@app.get("/thumbnails")
async def list_thumbnails(userid: str, page: int = 1, per_page: int = 20):
    """List thumbnails for a user with pagination"""
    per_page = min(per_page, 100)  # Limit max per page
    
    captures = database.get_captures_by_user(userid, page, per_page)
    
    base_url = settings.public_base_url
    thumbs = []
    for capture in captures:
        url = f"{base_url}/thumbnail/{userid}/{capture['id']}.jpeg"
        thumbs.append({
            'id': capture['id'],
            'time': capture['server_time'],
            'thumb': url
        })
    
    return {
        'page': page,
        'per_page': per_page,
        'items': thumbs
    }


@app.get("/image/{userid}/{image_id}.jpeg")
async def serve_full_image(userid: str, image_id: str):
    """Serve full-size image"""
    directory = f"storage/{userid}/truths"
    filename = f"{image_id}.jpeg"
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path, media_type='image/jpeg')


@app.get("/capture/{userid}/{image_id}")
async def get_capture(userid: str, image_id: str):
    """Get capture data by ID"""
    data = database.get_capture_by_id(userid, image_id)
    if not data:
        raise HTTPException(status_code=404, detail="Capture not found")
    return data


@app.get("/me/storage")
async def my_storage_report(current_user: dict = Depends(get_current_user), include_tmp: bool = False):
    """Get storage usage report for current user"""
    uid = current_user.get("email")
    
    # TODO: Implement storage calculation
    return {
        "userid": uid,
        "as_of": datetime.utcnow().isoformat(timespec='seconds') + "Z",
        "total": {"bytes": 0, "gb": 0.0},
        "breakdown": {},
        "include_tmp": include_tmp
    }


# Mount static files if enabled
if settings.enable_static_service:
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, debug=settings.debug)
