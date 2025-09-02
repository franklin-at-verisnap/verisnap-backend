"""
Geolocation and mapping utilities
"""
import requests
import json
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from astral import LocationInfo, Observer
from astral.sun import sun
# from timezonefinder import TimezoneFinder  # Temporarily disabled due to h3 dependency issues
import pytz
from dateutil import parser
from datetime import datetime, timezone

from config import settings


class GeolocationService:
    """Service for geolocation and mapping operations"""
    
    def __init__(self):
        self.api_key = settings.api_key
        # self.tf = TimezoneFinder()  # Temporarily disabled
    
    def reverse_geocode(self, lat: float, lng: float, timeout: int = 5) -> str:
        """Get formatted address from coordinates"""
        if not self.api_key:
            return self._reverse_geocode_local(lat, lng)
        
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={self.api_key}"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[reverse_geocode] error: {e}")
            return "nowhere"
        
        if data.get("status") == "OK" and data.get("results"):
            return data["results"][0].get("formatted_address", "nowhere")
        else:
            print(f"[reverse_geocode] API status not OK: {data.get('status')}")
            return "nowhere"
    
    def _reverse_geocode_local(self, latitude: float, longitude: float) -> str:
        """Fallback to local geocoding service"""
        geolocator = Nominatim(user_agent="Verisnap")
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        return location.address if location else "nowhere"
    
    def reverse_geocode_struct(self, lat: float, lng: float, timeout: int = 5) -> Dict[str, any]:
        """Get structured geocoding information"""
        if not self.api_key:
            return {"address": "nowhere", "loc": None, "location_type": None}
        
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={self.api_key}"
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[reverse_geocode_struct] error: {e}")
            return {"address": "nowhere", "loc": None, "location_type": None}
        
        if data.get("status") != "OK" or not data.get("results"):
            return {"address": "nowhere", "loc": None, "location_type": None}
        
        res = data["results"][0]
        loc = res["geometry"]["location"]
        ltype = res["geometry"].get("location_type")
        return {
            "address": res.get("formatted_address", "nowhere"),
            "loc": (loc["lat"], loc["lng"]),
            "location_type": ltype
        }
    
    def get_elevation(self, lat: float, lng: float) -> float:
        """Get elevation at coordinates"""
        if not self.api_key:
            return self._get_elevation_fallback(lat, lng)
        
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C{lng}&key={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data["status"] == "OK":
                return data["results"][0]["elevation"]
            else:
                print(f"Elevation API error: {data}")
                return self._get_elevation_fallback(lat, lng)
        except Exception as e:
            print(f"Elevation API error: {e}")
            return self._get_elevation_fallback(lat, lng)
    
    def _get_elevation_fallback(self, lat: float, lng: float) -> float:
        """Fallback elevation service"""
        api_url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lng}"
        try:
            response = requests.get(api_url)
            data = response.json()
            return data['results'][0]['elevation']
        except Exception as e:
            print(f"Fallback elevation error: {e}")
            return 0.0
    
    def places_establishment_within(self, lat: float, lng: float, radius_m: int = 20, timeout: int = 5) -> bool:
        """Check if there's an establishment within radius"""
        if not self.api_key:
            return False
        
        url = (
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            f"?location={lat},{lng}&radius={radius_m}&type=establishment&key={self.api_key}"
        )
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            d = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[places_establishment_within] error: {e}")
            return False
        
        if d.get("status") != "OK" or not d.get("results"):
            return False
        
        for place in d["results"]:
            p = place["geometry"]["location"]
            dist_m = geodesic((lat, lng), (p["lat"], p["lng"])).meters
            if dist_m <= radius_m:
                return True
        return False
    
    def is_daytime_at(self, lat: float, lon: float, iso_timestamp: str) -> bool:
        """Check if timestamp is during daylight hours at location"""
        # Parse timestamp
        dt = parser.isoparse(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Find timezone (simplified - use UTC for now)
        tz_name = "UTC"  # Simplified for now
        local_tz = pytz.timezone(tz_name)
        
        # Convert to local time
        local_dt = dt.astimezone(local_tz)
        
        # Compute sunrise & sunset
        obs = Observer(latitude=lat, longitude=lon)
        s = sun(observer=obs, date=local_dt.date(), tzinfo=local_tz)
        
        return s["sunrise"] <= local_dt <= s["sunset"]
    
    def get_local_pressures(self, lat: float, lon: float, timeout: int = 5) -> Dict[str, Optional[float]]:
        """Get local atmospheric pressure data"""
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current=pressure_msl,surface_pressure"
        )
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            cur = d.get('current', {})
            return {
                'pressure_msl': cur.get('pressure_msl'),
                'surface_pressure': cur.get('surface_pressure')
            }
        except Exception as e:
            print(f"[get_local_pressures] error: {e}")
            return {'pressure_msl': None, 'surface_pressure': None}
