"""
Image processing utilities
"""
import os
import base64
import hashlib
import imagehash
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ExifTags
from io import BytesIO
from typing import Tuple, Optional
import torch
import clip
import joblib
import warnings

# Optional imports that might fail
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, some features will be disabled")

from config import settings

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names.*"
)


class ImageProcessor:
    """Image processing utilities for verification and analysis"""
    
    def __init__(self):
        self.device = self._get_device()
        self.model, self.preprocess = self._load_clip_model()
        self.midas = self._load_midas_model()
        self.p2p_classifier = self._load_p2p_classifier()
        self.midas_transform = self._get_midas_transform()
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if settings.device != "auto":
            return settings.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_clip_model(self) -> Tuple[torch.nn.Module, callable]:
        """Load CLIP model for image analysis"""
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess
    
    def _load_midas_model(self) -> torch.nn.Module:
        """Load MiDaS model for depth estimation"""
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas.to(self.device).eval()
        return midas
    
    def _load_p2p_classifier(self) -> object:
        """Load photo-of-photo classifier"""
        return joblib.load(settings.model_path)
    
    def _get_midas_transform(self) -> callable:
        """Get MiDaS preprocessing transform"""
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
        
        M_ID, M_SD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        return Compose([
            Resize((384, 384), interpolation=Image.BILINEAR),
            ToTensor(),
            Normalize(mean=M_ID, std=M_SD),
        ])
    
    def decode_base64_image(self, base64_string: str) -> bytes:
        """Decode base64 image string"""
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        return base64.b64decode(base64_string)
    
    def calculate_sha256(self, image_path: str) -> str:
        """Calculate SHA256 hash of image file"""
        sha256_hash = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def calculate_average_hash(self, image_path: str) -> str:
        """Calculate average hash of image"""
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            return str(imagehash.average_hash(img))
    
    def generate_thumbnail(self, source_path: str, thumb_path: str, size: Tuple[int, int] = (200, 200)):
        """Generate thumbnail from source image"""
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        with Image.open(source_path) as img:
            img.thumbnail(size)
            img.save(thumb_path, format="JPEG")
    
    def add_watermark(self, input_path: str, output_path: str, watermark_text: str):
        """Add watermark to image with adaptive contrast"""
        photo = Image.open(input_path)
        
        # Handle EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = dict(photo._getexif().items())
            if exif[orientation] == 3:
                photo = photo.rotate(180, expand=True)
            elif exif[orientation] == 6:
                photo = photo.rotate(270, expand=True)
            elif exif[orientation] == 8:
                photo = photo.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        
        drawing = ImageDraw.Draw(photo)
        
        # Font configuration
        font_path = '/System/Library/Fonts/Arial.ttf'
        base_main_size = 70
        base_sub_size = 40
        font_main = ImageFont.truetype(font_path, int(base_main_size * 1.4))
        font_sub = ImageFont.truetype(font_path, int(base_sub_size * 1.4))
        
        # Calculate text dimensions
        text_width = drawing.textlength(watermark_text, font=font_main)
        text_height = font_main.getbbox(watermark_text)[3]
        
        # Position watermark
        margin_x, margin_y = 110, 120
        x = photo.width - text_width - margin_x
        y = photo.height - text_height - margin_y
        
        # Determine local brightness for contrast
        arr = np.array(photo.convert('RGB'))
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x2 = min(photo.width, int(x + text_width))
        y2 = min(photo.height, int(y + text_height))
        
        if x2 > x0 and y2 > y0:
            roi = arr[y0:y2, x0:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            v_mean = float(np.mean(hsv[:, :, 2])) / 255.0
        else:
            v_mean = 0.5
        
        # Choose colors for contrast
        if v_mean >= 0.6:
            fill_col = (0, 0, 0)
            stroke_col = (255, 255, 255)
        else:
            fill_col = (255, 255, 255)
            stroke_col = (0, 0, 0)
        
        # Draw main watermark
        stroke_w = max(2, font_main.size // 24)
        drawing.text((x, y), watermark_text, fill=fill_col, font=font_main,
                     stroke_width=stroke_w, stroke_fill=stroke_col)
        
        # Draw "by Verisnap" subtitle
        sub_text = "by Verisnap"
        text_width_sub = drawing.textlength(sub_text, font=font_sub)
        text_height_sub = font_sub.getbbox(sub_text)[3]
        
        x1 = x + int(135 * 1.4)
        y1 = y + int(64 * 1.4)
        
        # Check brightness for subtitle area
        x0s = max(0, int(x1))
        y0s = max(0, int(y1))
        x2s = min(photo.width, int(x1 + text_width_sub))
        y2s = min(photo.height, int(y1 + text_height_sub))
        
        if x2s > x0s and y2s > y0s:
            roi2 = arr[y0s:y2s, x0s:x2s]
            hsv2 = cv2.cvtColor(roi2, cv2.COLOR_RGB2HSV)
            v_mean2 = float(np.mean(hsv2[:, :, 2])) / 255.0
        else:
            v_mean2 = v_mean
        
        if v_mean2 >= 0.6:
            fill_col2 = (0, 0, 0)
            stroke_col2 = (255, 255, 255)
        else:
            fill_col2 = (255, 255, 255)
            stroke_col2 = (0, 0, 0)
        
        stroke_w2 = max(2, font_sub.size // 24)
        drawing.text((x1, y1), sub_text, fill=fill_col2, font=font_sub,
                     stroke_width=stroke_w2, stroke_fill=stroke_col2)
        
        photo.save(output_path)
    
    def detect_picture_of_picture(self, image_path: str) -> Tuple[bool, float, float, float, float, float]:
        """
        Detect if image is a photo-of-photo using ML features
        
        Returns:
            Tuple of (is_p2p, planar_ratio, variance, edge_density, border_edge_density, probability)
        """
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        
        # Depth-based features using MiDaS
        inp = self.midas_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.size[::-1],
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()
        
        depth = (pred - pred.min()) / (pred.max() - pred.min())
        h, w = depth.shape
        roi_frac = 0.5
        rh, rw = int(h*roi_frac), int(w*roi_frac)
        y0, x0 = (h-rh)//2, (w-rw)//2
        roi = depth[y0:y0+rh, x0:x0+rw]
        
        eps = 0.05
        planar_ratio = (np.abs(roi - np.median(roi)) < eps).sum() / roi.size
        variance = float(np.var(roi))
        
        # Global edge density
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200) > 0
        edge_density = edges.sum() / edges.size
        
        # Border edge density
        margin = int(0.05 * min(h, w))
        mask = np.zeros_like(edges, bool)
        mask[:margin, :] = True
        mask[-margin:, :] = True
        mask[:, :margin] = True
        mask[:, -margin:] = True
        border_edge_density = edges[mask].sum() / mask.sum()
        
        # Use classifier
        features = [[planar_ratio, variance, edge_density, border_edge_density]]
        pred = self.p2p_classifier.predict(features)[0]
        prob = self.p2p_classifier.predict_proba(features)[0, 1]
        
        return bool(pred), planar_ratio, variance, edge_density, border_edge_density, prob
