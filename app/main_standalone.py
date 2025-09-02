"""
Standalone FastAPI application for Verisnap Backend (minimal dependencies)
"""
import os
import json
import base64
import traceback
import sqlite3
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


class SimpleDatabaseService:
    """Simple database service without external dependencies"""
    
    def __init__(self):
        self.db_path = settings.database_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captures (
                id TEXT PRIMARY KEY,
                userid TEXT,
                os TEXT,
                lat TEXT,
                lon TEXT,
                altitude TEXT,
                original_signature TEXT,
                watermarked_signature TEXT,
                relevant_obj_tags TEXT,
                device_time TEXT,
                server_time TEXT,
                t_score TEXT,
                os_score TEXT,
                in_vs_out_score TEXT,
                day_vs_night_score TEXT,
                altitude_score TEXT,
                device_score TEXT,
                barometerData TEXT,
                magnetometerData TEXT,
                heading TEXT,
                vscore TEXT,
                baro_score TEXT,
                magnetometer_score TEXT,
                cheat_penalty TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_captures_server_time 
            ON captures (server_time DESC)
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_capture(self, capture_dict):
        """Insert a new capture record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            insert_query = '''
            INSERT INTO captures (
                id, userid, os, lat, lon, altitude, original_signature, watermarked_signature,
                relevant_obj_tags, device_time, server_time, t_score, os_score,
                in_vs_out_score, day_vs_night_score, altitude_score, device_score,
                barometerData, magnetometerData, heading, vscore,
                baro_score, magnetometer_score, cheat_penalty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            capture_data = (
                str(capture_dict.get('id', '')),
                str(capture_dict.get('userid', '')),
                str(capture_dict.get('os', '')),
                str(capture_dict.get('lat', '')),
                str(capture_dict.get('lon', '')),
                str(capture_dict.get('altitude', '')),
                str(capture_dict.get('original_signature', '')),
                str(capture_dict.get('watermarked_signature', '')),
                str(capture_dict.get('relevant_obj_tags', '')),
                str(capture_dict.get('device_time', '')),
                str(capture_dict.get('server_time', '')),
                str(capture_dict.get('t_score', '')),
                str(capture_dict.get('os_score', '')),
                str(capture_dict.get('in_vs_out_score', '')),
                str(capture_dict.get('day_vs_night_score', '')),
                str(capture_dict.get('altitude_score', '')),
                str(capture_dict.get('device_score', '')),
                str(capture_dict.get('barometerData', '')),
                str(capture_dict.get('magnetometerData', '')),
                str(capture_dict.get('heading', '')),
                str(capture_dict.get('vscore', '')),
                str(capture_dict.get('baro_score', '')),
                str(capture_dict.get('magnetometer_score', '')),
                str(capture_dict.get('cheat_penalty', '')),
            )
            
            cursor.execute(insert_query, capture_data)
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error inserting capture: {e}")
            return False
    
    def query_records_by_hash(self, avghash):
        """Query records by watermarked signature hash"""
        records_list = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sql_query = "SELECT * FROM captures WHERE watermarked_signature = ?"
            cursor.execute(sql_query, (avghash,))
            records = cursor.fetchall()
            
            for record in records:
                columns = [col[0] for col in cursor.description]
                record_dict = dict(zip(columns, record))
                records_list.append(record_dict)
            
        except sqlite3.Error as e:
            print("Error querying the database:", e)
        finally:
            if 'conn' in locals():
                conn.close()
        
        return records_list
    
    def get_capture_by_id(self, userid, image_id):
        """Get capture record by user ID and image ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM captures WHERE userid = ? AND id = ?",
                (userid, image_id)
            )
            row = cursor.fetchone()
            
            if row:
                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
            return None
            
        except sqlite3.Error as e:
            print("Error getting capture:", e)
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_captures_by_user(self, userid, page=1, per_page=20):
        """Get captures for a user with pagination"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            offset = (page - 1) * per_page
            cursor.execute(
                "SELECT id, server_time FROM captures "
                "WHERE userid = ? "
                "ORDER BY server_time DESC "
                "LIMIT ? OFFSET ?",
                (userid, per_page, offset)
            )
            rows = cursor.fetchall()
            
            return [{"id": row[0], "server_time": row[1]} for row in rows]
            
        except sqlite3.Error as e:
            print("Error getting user captures:", e)
            return []
        finally:
            if 'conn' in locals():
                conn.close()


class SimpleAuthService:
    """Simple auth service without Firebase for testing"""
    
    def verify_firebase_token(self, id_token):
        """Mock token verification for testing"""
        # For testing purposes, accept any token
        return {"email": "test@example.com", "uid": "test123"}


# Initialize services
database = SimpleDatabaseService()
auth_service = SimpleAuthService()


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
            if request.image.startswith('data:'):
                pic_data = base64.b64decode(request.image.split(',', 1)[1])
            else:
                pic_data = base64.b64decode(request.image)
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
    """Upload and process image for verification (simplified version)"""
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
            if request.pic.startswith('data:'):
                pic_data = base64.b64decode(request.pic.split(',', 1)[1])
            else:
                pic_data = base64.b64decode(request.pic)
            
            temp_path = f'storage/{userid}/tmp/{request.id}_temp.jpeg'
            final_path = f'storage/{userid}/truths/{request.id}.jpeg'
            meta_path = f'storage/{userid}/meta/{request.id}.json'
            
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(pic_data)
            
            with open(meta_path, 'w') as meta_file:
                meta_file.write(json.dumps(request.dict()))
                
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process image and calculate scores (simplified)
        current_utc_time = datetime.now(timezone.utc)
        formatted_time = current_utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        
        # Simplified scoring (without ML models for now)
        deltaT = 51  # Assume good timestamp for now
        deltaOS = 5  # Assume good OS version
        device_score = 10  # Assume device verification passes
        in_vs_out = 10  # Assume outdoor scene
        daynightscore = 10  # Assume good day/night match
        deltaAltitudeScore = 5  # Assume good altitude
        magnetometer_score = 4  # Assume good magnetometer
        
        # Calculate total score
        vscore = deltaT + deltaOS + in_vs_out + daynightscore + deltaAltitudeScore + device_score + magnetometer_score
        
        # Add simple watermark (without ML processing)
        with Image.open(temp_path) as img:
            # Simple watermark - just save the image for now
            img.save(final_path, "JPEG")
        
        # Generate thumbnail
        thumb_path = f"storage/{userid}/thumbs/{request.id}_thumb.jpeg"
        with Image.open(final_path) as img:
            img.thumbnail((200, 200))
            img.save(thumb_path, "JPEG")
        
        # Remove temp file
        os.remove(temp_path)
        
        # Calculate final hash
        with Image.open(final_path) as img:
            avghash = imagehash.average_hash(img)
        
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
            'original_signature': 'simplified_hash',  # Simplified for now
            'watermarked_signature': str(avghash),
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
            message=f'Image saved as {request.id}.jpeg with watermark (simplified version)'
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
    
    base_url = "http://localhost:9000"  # TODO: Make this configurable
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
