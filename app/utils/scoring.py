"""
Scoring and verification utilities
"""
import json
import math
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
import clip
from dateutil import parser
from datetime import datetime, timezone

from .image_processing import ImageProcessor
from .geolocation import GeolocationService
from config import settings


class ScoringService:
    """Service for calculating verification scores"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.geolocation = GeolocationService()
        self.device = self.image_processor.device
        self.model = self.image_processor.model
        self.preprocess = self.image_processor.preprocess
    
    def _clip_prob(self, img_tensor: torch.Tensor, prompts: List[str]) -> np.ndarray:
        """Get CLIP probabilities for prompts"""
        with torch.no_grad():
            logits, _ = self.model(img_tensor, clip.tokenize(prompts).to(self.device))
            return logits.softmax(dim=-1).cpu().numpy()[0]
    
    def context_overrides_indoor(self, pil_img: Image.Image, margin_frac: float = 0.06, border_votes: int = 2) -> bool:
        """Check if context suggests indoor scene with outdoor center (framed prints/TVs)"""
        W, H = pil_img.size
        mW, mH = max(8, int(W*margin_frac)), max(8, int(H*margin_frac))
        
        # Define regions
        center = pil_img.crop((mW, mH, W - mW, H - mH))
        top = pil_img.crop((0, 0, W, mH))
        bottom = pil_img.crop((0, H - mH, W, H))
        left = pil_img.crop((0, 0, mW, H))
        right = pil_img.crop((W - mW, 0, W, H))
        
        def prob_io(pil):
            x = self.preprocess(pil).unsqueeze(0).to(self.device)
            p = self._clip_prob(x, ["an indoor scene", "an outdoor scene"])
            return float(p[0]), float(p[1])  # p_indoor, p_outdoor
        
        p_center_in, p_center_out = prob_io(center)
        
        votes_in = 0
        for r in (top, bottom, left, right):
            p_in, p_out = prob_io(r)
            votes_in += (p_in > p_out)
        
        return (votes_in >= border_votes) and (p_center_out > p_center_in)
    
    def has_framed_rectangle(self, np_img: np.ndarray, min_area_frac: float = 0.08, eps_ratio: float = 0.02) -> bool:
        """Detect large inner convex rectangles (frames/TVs)"""
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        H, W = gray.shape
        img_area = W * H
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_ratio*peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area / img_area >= min_area_frac:
                    return True
        return False
    
    def clip_displayed_media_prob(self, img_tensor: torch.Tensor) -> Dict[str, float]:
        """Get probabilities for content being on screen/framed vs real scene"""
        prompts = [
            "a real outdoor scene",
            "a view through a window",
            "a photo displayed on a TV screen",
            "a photo displayed on a computer monitor",
            "a photo displayed on a phone screen",
            "a framed photo hanging on a wall"
        ]
        
        probs = self._clip_prob(img_tensor, prompts)
        
        return {
            "real": float(probs[0]),
            "window": float(probs[1]),
            "screen": float(probs[2] + probs[3] + probs[4]),
            "frame": float(probs[5])
        }
    
    def sky_fraction_top(self, np_img: np.ndarray, top_frac: float = 0.5) -> float:
        """Estimate sky coverage in top portion of image"""
        h, w, _ = np_img.shape
        top = np_img[:int(h*top_frac), :, :]
        hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1] / 255.0
        V = hsv[:, :, 2] / 255.0
        
        # Blue-ish sky detection
        sky = ((H >= 95) & (H <= 135) & (S > 0.15) & (V > 0.45)) | \
              ((H >= 90) & (H <= 130) & (S > 0.10) & (V > 0.65))
        return float(np.mean(sky))
    
    def pop_score_from_exif(self, exif_md_or_json: Any) -> Dict[str, Any]:
        """Compute Photo-of-Photo score from EXIF metadata"""
        # Parse input
        if isinstance(exif_md_or_json, str):
            try:
                md = json.loads(exif_md_or_json)
            except Exception:
                return {"score": 0.0, "decision": "unlikely_pop", "features": {}, "debug": {"error": "bad_json"}}
        elif isinstance(exif_md_or_json, dict):
            md = exif_md_or_json
        else:
            return {"score": 0.0, "decision": "unlikely_pop", "features": {}, "debug": {"error": "bad_input_type"}}
        
        def get(d, path, default=None):
            cur = d
            for k in path.split('.'):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur
        
        def g(path, default=None):
            root = md.get("imageMetadata", md)
            return get(root, path, default)
        
        # Extract values
        px = g("{Exif}.PixelXDimension") or g("PixelWidth")
        py = g("{Exif}.PixelYDimension") or g("PixelHeight")
        subject_area = g("{Exif}.SubjectArea") or []
        flash = g("{Exif}.Flash") or 0
        comp = g("{Exif}.CompositeImage") or 0
        bv = g("{Exif}.BrightnessValue")
        
        iso_arr = g("{Exif}.ISOSpeedRatings")
        iso = None
        if isinstance(iso_arr, list) and iso_arr:
            iso = iso_arr[0]
        elif isinstance(iso_arr, (int, float)):
            iso = iso_arr
        
        exposure_time = g("{Exif}.ExposureTime")
        fnum = g("{Exif}.FNumber")
        orient = g("{TIFF}.Orientation") or g("Orientation")
        gvec = g("{MakerApple}.8", [None, None, None])
        gx, gy, gz = (gvec + [None, None, None])[:3]
        
        # Calculate features
        flatness = 0.0
        if isinstance(gy, (int, float)):
            if abs(gy) < 0.2:
                flatness = max(0.0, min(1.0, (0.2 - abs(gy)) / 0.2))
        
        af_tight_center = 0.0
        if px and py and isinstance(subject_area, list) and len(subject_area) >= 4:
            cx, cy, w, h = subject_area[:4]
            try:
                area_ratio = (w * h) / (px * py)
                centered = (abs(cx - px/2) <= 0.12*px) and (abs(cy - py/2) <= 0.12*py)
                if area_ratio < 0.08 and centered:
                    af_tight_center = 1.0
            except Exception:
                pass
        
        hdr_small = 1.0 if (comp == 2 and af_tight_center == 1.0) else 0.0
        
        flash_suspicious = 0.0
        if isinstance(flash, int) and (flash != 0) and isinstance(bv, (int, float)):
            if bv > 5.0:
                flash_suspicious = 1.0
        
        indoor_pattern = 0.0
        if isinstance(iso, (int, float)) and isinstance(exposure_time, (int, float)) and isinstance(fnum, (int, float)):
            if (150 <= iso <= 800) and (1/125 <= exposure_time <= 1/30) and (1.5 <= fnum <= 1.7):
                indoor_pattern = 1.0
        
        orientation_incoherent = 0.0
        if isinstance(orient, int) and isinstance(gy, (int, float)):
            if orient in (6, 8) and abs(gy) < 0.4:
                orientation_incoherent = 0.5
            if orient == 1 and abs(gy) > 0.9:
                orientation_incoherent = 0.5
        
        # Calculate score
        score = (
            0.55 * flatness +
            0.25 * af_tight_center +
            0.08 * hdr_small +
            0.06 * flash_suspicious +
            0.04 * indoor_pattern +
            0.02 * orientation_incoherent
        )
        score = max(0.0, min(1.0, float(score)))
        
        # Decision
        if score >= 0.65:
            decision = "likely_pop"
        elif score >= 0.45:
            decision = "borderline"
        else:
            decision = "unlikely_pop"
        
        return {
            "score": score,
            "decision": decision,
            "features": {
                "flatness": flatness,
                "af_tight_center": af_tight_center,
                "hdr_small": hdr_small,
                "flash_suspicious": flash_suspicious,
                "indoor_pattern": indoor_pattern,
                "orientation_incoherent": orientation_incoherent,
            },
            "debug": {
                "px": px, "py": py, "subject_area": subject_area,
                "flash": flash, "bv": bv, "iso": iso,
                "exposure_time": exposure_time, "fnumber": fnum,
                "orientation": orient, "gvec": [gx, gy, gz],
            }
        }
    
    def calculate_magnetometer_score(self, data: Dict[str, Any], exif_md: Optional[Dict] = None) -> int:
        """Calculate magnetometer verification score"""
        heading_val = data.get('heading')
        raw_mag = data.get('magnetometerData')
        
        if raw_mag is None:
            print("Error: Missing magnetometer data")
            return -4
        
        # Parse magnetometer data
        if isinstance(raw_mag, str):
            try:
                raw_mag = json.loads(raw_mag)
            except json.JSONDecodeError:
                print("Error: Invalid magnetometer JSON data")
                return -4
        
        # Extract components
        try:
            x, y, z = float(raw_mag['x']), float(raw_mag['y']), float(raw_mag['z'])
        except (KeyError, ValueError, TypeError):
            print("Error: Magnetometer data missing or invalid x, y, or z components")
            return -4
        
        # Convert units if needed
        units = raw_mag.get('units')
        x, y, z = self._convert_to_uT(x, y, z, units)
        
        # Auto-detect units if not provided
        mag_tmp = math.sqrt(x*x + y*y + z*z)
        if units is None:
            if mag_tmp > 2000:  # likely mG
                x, y, z = x * 0.1, y * 0.1, z * 0.1
            elif mag_tmp < 1.0:  # likely mT
                x, y, z = x * 1000.0, y * 1000.0, z * 1000.0
        
        # Calculate magnitude
        mag = math.sqrt(x*x + y*y + z*z)
        expected_mag = 50.0  # Earth's field magnitude target (µT)
        mag_tolerance = 25.0
        
        # Score magnitude
        mag_factor = max(0.0, 1.0 - abs(mag - expected_mag) / mag_tolerance)
        mag_score = mag_factor * 2.0
        
        # Score heading if device is flat enough
        heading_score = 0.0
        if heading_val is not None:
            gy = None
            if isinstance(exif_md, dict):
                root = exif_md.get("imageMetadata", exif_md)
                gvec = root.get("{MakerApple}.8", [None, None, None])
                if isinstance(gvec, list) and len(gvec) >= 2:
                    try:
                        gy = float(gvec[1])
                    except (TypeError, ValueError):
                        gy = None
            
            horizontal_ok = (gy is not None and abs(gy) < 0.35)
            
            if horizontal_ok:
                try:
                    heading = float(heading_val)
                    heading_mag = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
                    diff = abs(heading_mag - heading)
                    diff = min(diff, 360.0 - diff)
                    
                    heading_tolerance = 30.0
                    heading_factor = max(0.0, 1.0 - diff / heading_tolerance)
                    heading_score = heading_factor * 2.0
                except (ValueError, TypeError):
                    heading_score = 0.0
        
        total = mag_score + heading_score
        return int(round(total))
    
    def _convert_to_uT(self, x: float, y: float, z: float, units: Optional[str]) -> Tuple[float, float, float]:
        """Convert magnetometer components to microtesla"""
        if not units:
            return x, y, z
        
        u = units.strip().lower()
        if u in {"ut", "µt", "microtesla", "micro-tesla"}:
            f = 1.0
        elif u in {"mt", "millitesla", "milli-tesla"}:
            f = 1000.0
        elif u in {"g", "gauss"}:
            f = 100.0
        elif u in {"mg", "milligauss", "milli-gauss"}:
            f = 0.1
        else:
            f = 1.0
        
        return x * f, y * f, z * f
    
    def compare_timestamp_with_current_utc(self, timestamp_str: str) -> int:
        """Compare timestamp with current UTC time"""
        ts = parser.isoparse(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        current_utc = datetime.now(timezone.utc)
        diff_ms = abs((ts - current_utc).total_seconds() * 1000.0)
        return 51 if diff_ms <= 30_000 else 0
    
    def compare_version_strings(self, v1: str, v2: str) -> int:
        """Compare version strings"""
        def parts(v): 
            return [int(''.join(filter(str.isdigit, p)) or 0) for p in v.split('.')]
        
        a, b = parts(v1), parts(v2)
        for x, y in zip(a, b):
            if x != y: 
                return 1 if x > y else -1
        return (1 if len(a) > len(b) and any(a[len(b):]) else
                -1 if len(b) > len(a) and any(b[len(a):]) else 0)
    
    def baro_altitude_from_pressure(self, pressure_hpa: float, sea_level_hpa: float) -> Optional[float]:
        """Estimate altitude from barometric pressure"""
        try:
            P = float(pressure_hpa)
            P0 = float(sea_level_hpa)
            if P <= 0 or P0 <= 0:
                return None
            return 44330.0 * (1.0 - (P / P0) ** (1.0 / 5.255))
        except Exception:
            return None
