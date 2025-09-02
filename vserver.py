import json
from flask import Flask, request, jsonify, make_response
from PIL import Image, ImageDraw, ImageFont, ExifTags
import base64
import os
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime, timezone
import torch
import clip
from io import BytesIO

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

import geopy
from geopy.geocoders import Nominatim
from astral import LocationInfo
from astral.sun import sun
import pytz
from shapely.geometry import Point, Polygon
import requests
import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
#from flask_compress import Compress
import gzip
import traceback
import sqlite3
import hashlib
import imagehash
import uuid
from flask import send_from_directory
from flask import Flask
from flask_cors import CORS
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import joblib
import warnings
from astral import Observer
from astral.sun import sun
from timezonefinder import TimezoneFinder
import pytz
from dateutil import parser 
import math 
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from functools import wraps
from flask import request, jsonify
from geopy.distance import geodesic
import csv

# load your service-account JSON
cred = credentials.Certificate("verisnap-poc-firebase-adminsdk-fbsvc-d8f2304cdb.json")
firebase_admin.initialize_app(cred)

# Suppress the “X does not have valid feature names” warning from LogisticRegression
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names.*"
)

# at module‐level, once:
clf = joblib.load("models/p2p_classifier.joblib")


# 1️⃣ Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  
midas.to(device).eval()

from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# MiDaS normalization stats
M_ID, M_SD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

midas_transform = Compose([
    Resize((384, 384), interpolation=Image.BILINEAR),
    ToTensor(),
    Normalize(mean=M_ID, std=M_SD),
])

app = Flask(__name__, static_folder="static", static_url_path="")
# Allow all origins (you can lock this down to your iOS app's domain if needed)
CORS(app, resources={r"/thumbnails": {"origins": "*"},
                     r"/thumbnail/*": {"origins": "*"}})


# Google elevation API key (loaded from .env)
api_key = os.getenv("API_KEY")

# Apple JWT configuration (loaded from .env)
kid = os.getenv("APPLE_KID")
team_id = os.getenv("APPLE_TEAM_ID")
private_key_path = os.getenv("PRIVATE_KEY_PATH")

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(args)
        auth_header = request.headers.get("Authorization", None)
        if not auth_header or not auth_header.startswith("Bearer "):
            msg = "Missing or malformed Authorization header"
            print(msg)  # prints to your server console
            return jsonify({"error": "Unauthorized", "message": msg}), 401

        id_token = auth_header.split()[1]
        try:
            # verifies signature, expiration, etc.
            decoded = firebase_auth.verify_id_token(id_token)
            request.user = decoded
        except Exception as e:
            msg = f"Token verification failed: {e}"
            print(msg)  # prints the exception message
            return jsonify({"error": "Unauthorized", "message": msg}), 401

        return f(*args, **kwargs)
    return wrapper

def _clip_prob(img_tensor, prompts):
    with torch.no_grad():
        logits, _ = model(img_tensor, clip.tokenize(prompts).to(device))
        return logits.softmax(dim=-1).cpu().numpy()[0]

def context_overrides_indoor(pil_img, margin_frac=0.06, border_votes=2):
    """
    Returns True if at least `border_votes` of the 4 borders look indoor
    while the center looks outdoor. Strong signal for framed prints/TVs.
    """
    W, H = pil_img.size
    mW, mH = max(8, int(W*margin_frac)), max(8, int(H*margin_frac))

    # Regions
    center = pil_img.crop((mW, mH, W - mW, H - mH))
    top    = pil_img.crop((0, 0, W, mH))
    bottom = pil_img.crop((0, H - mH, W, H))
    left   = pil_img.crop((0, 0, mW, H))
    right  = pil_img.crop((W - mW, 0, mW, H))

    def prob_io(pil):
        x = preprocess(pil).unsqueeze(0).to(device)
        p = _clip_prob(x, ["an indoor scene", "an outdoor scene"])
        return float(p[0]), float(p[1])  # p_indoor, p_outdoor

    p_center_in, p_center_out = prob_io(center)

    votes_in = 0
    for r in (top, bottom, left, right):
        p_in, p_out = prob_io(r)
        votes_in += (p_in > p_out)

    return (votes_in >= border_votes) and (p_center_out > p_center_in)


def has_framed_rectangle(np_img, min_area_frac=0.08, eps_ratio=0.02):
    """
    Cheap rectangle cue for frames/TVs. True if we detect a large inner convex quad.
    """
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


def reverse_geocode_struct(lat, lng, api_key, timeout=5):
    """
    Returns a SMALL struct with address, location (lat,lng), and location_type.
    On any failure returns {"address":"nowhere", "loc": None, "location_type": None}
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[reverse_geocode_struct] error: {e}")
        return {"address":"nowhere","loc":None,"location_type":None}

    if data.get("status") != "OK" or not data.get("results"):
        return {"address":"nowhere","loc":None,"location_type":None}

    res = data["results"][0]
    loc = res["geometry"]["location"]
    ltype = res["geometry"].get("location_type")
    return {
        "address": res.get("formatted_address","nowhere"),
        "loc": (loc["lat"], loc["lng"]),
        "location_type": ltype
    }

def places_establishment_within(lat, lng, api_key, radius_m=20, timeout=5):
    """
    True if there is an 'establishment' within radius_m of (lat,lng), using geodesic meters.
    """
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lng}&radius={radius_m}&type=establishment&key={api_key}"
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

def clip_displayed_media_prob(img_tensor):
    """
    Returns probabilities for content being on a screen / framed photo vs real scene.
    """
    prompts = [
        "a real outdoor scene",                 # 0
        "a view through a window",              # 1
        "a photo displayed on a TV screen",     # 2
        "a photo displayed on a computer monitor", # 3
        "a photo displayed on a phone screen",  # 4
        "a framed photo hanging on a wall"      # 5
    ]
    with torch.no_grad():
        logits, _ = model(img_tensor, clip.tokenize(prompts).to(device))
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    p_real   = float(probs[0])
    p_window = float(probs[1])
    p_screen = float(probs[2] + probs[3] + probs[4])
    p_frame  = float(probs[5])
    return {"real": p_real, "window": p_window, "screen": p_screen, "frame": p_frame}

def sky_fraction_top(np_img, top_frac=0.5):
    """
    Estimate sky coverage in the top portion of the image.
    Stricter: require blue-ish hue + some saturation + decent brightness.
    (OpenCV HSV has H in [0..179]. Blue ~ [95..135].)
    """
    h, w, _ = np_img.shape
    top = np_img[:int(h*top_frac), :, :]
    hsv = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]             # 0..179
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    # Blue-ish sky only (no generic "bright & desaturated" which matched posters/paper)
    sky = ((H >= 95) & (H <= 135) & (S > 0.15) & (V > 0.45)) \
          | ((H >= 90) & (H <= 130) & (S > 0.10) & (V > 0.65))
    return float(np.mean(sky))

def decide_indoor_outdoor(temp_path, lat, lon, iso_ts, api_key, exif_md=None, barometer=None):
    debug = {}
    adjustments = {'magnetometer_cap': None, 'cheat_penalty': 0}
    in_vs_out = 0
    daynight = 0

    # --- load once ---
    pil_img = Image.open(temp_path).convert("RGB")
    np_img  = np.array(pil_img)
    img     = preprocess(pil_img).unsqueeze(0).to(device)

    # --- CLIP indoor/outdoor ---
    txt_io = clip.tokenize(["an indoor scene", "an outdoor scene"]).to(device)
    with torch.no_grad():
        l_io, _ = model(img, txt_io)
        p_io = l_io.softmax(dim=-1).cpu().numpy()[0]
    p_indoor, p_outdoor = float(p_io[0]), float(p_io[1])
    debug['p_indoor'] = p_indoor; debug['p_outdoor'] = p_outdoor

    # --- context + frame cues ---
    try:
        context_indoor = context_overrides_indoor(pil_img, margin_frac=0.06, border_votes=2)
    except Exception:
        context_indoor = False
    try:
        frame_present  = has_framed_rectangle(np_img, min_area_frac=0.08, eps_ratio=0.02)
    except Exception:
        frame_present = False
    debug['context_indoor_override'] = context_indoor
    debug['frame_present'] = frame_present

    # --- day/night CLIP ---
    if p_outdoor >= 0.6:
        txt_dn = clip.tokenize(["an outdoor scene in daylight", "an outdoor scene at night"]).to(device)
    elif p_indoor >= 0.6:
        txt_dn = clip.tokenize(["an indoor scene during the day", "an indoor scene at night"]).to(device)
    else:
        txt_dn = clip.tokenize(["a photo of a daylight scene", "a photo of a nighttime scene"]).to(device)
    with torch.no_grad():
        l_dn, _ = model(img, txt_dn)
        p_dn = l_dn.softmax(dim=-1).cpu().numpy()[0]
    p_day, p_night = float(p_dn[0]), float(p_dn[1])
    debug['p_day'] = p_day; debug['p_night'] = p_night

    local_is_day = is_daytime_at(lat, lon, iso_ts)
    debug['local_is_day'] = local_is_day

    # --- CLIP "displayed media" probe (screen / monitor / phone / framed photo) ---
    disp = clip_displayed_media_prob(img)
    debug['p_real']   = disp['real']
    debug['p_window'] = disp['window']
    debug['p_screen'] = disp['screen']
    debug['p_frame']  = disp['frame']
    # TIGHTER: make "screen-like" harder to trip
    screen_like = (disp['screen'] >= 0.40) and (disp['screen'] - max(disp['real'], disp['window'])) >= 0.18
    debug['screen_like'] = screen_like

    # --- luminance + specular ---
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    V = hsv[:, :, 2] / 255.0
    S = hsv[:, :, 1] / 255.0
    median_v  = float(np.median(V))
    low_light = median_v < 0.25
    specular_ratio = float(np.mean((V > 0.985) & (S < 0.25)))  # glossy hotspots
    debug['median_v'] = median_v
    debug['low_light'] = low_light
    debug['specular_ratio'] = specular_ratio

    # Initialize cheat reasons accumulator
    cheat_reasons = []

    # Defer displayed-media anti-cheat until after we have sky and EXIF PoP

    # --- EXIF priors ---
    flash = None; bv = None; iso = None; exposure_time = None; fnum = None
    gx = gy = gz = None
    if isinstance(exif_md, dict):
        root = exif_md.get("imageMetadata", exif_md)
        flash = root.get("{Exif}.Flash", root.get("Flash"))
        bv    = root.get("{Exif}.BrightnessValue", root.get("BrightnessValue"))
        iso_a = root.get("{Exif}.ISOSpeedRatings")
        if isinstance(iso_a, list) and iso_a: iso = iso_a[0]
        elif isinstance(iso_a, (int, float)): iso = iso_a
        exposure_time = root.get("{Exif}.ExposureTime", root.get("ExposureTime"))
        fnum = root.get("{Exif}.FNumber")
        gvec = root.get("{MakerApple}.8", [None, None, None])
        gx, gy, gz = (gvec + [None, None, None])[:3]

    exif_flatness = 0.0
    if isinstance(gy, (int, float)):
        exif_flatness = (0.2 - abs(gy)) / 0.2 if abs(gy) < 0.2 else 0.0
        exif_flatness = max(0.0, min(1.0, float(exif_flatness)))

    indoor_exposure = 0.0
    if isinstance(iso, (int,float)) and isinstance(exposure_time, (int,float)) and isinstance(fnum, (int,float)):
        if (150 <= iso <= 800) and (1/125 <= exposure_time <= 1/30) and (1.5 <= fnum <= 1.8):
            indoor_exposure = 1.0

    debug['exif_flatness'] = exif_flatness
    debug['exif_indoor_exposure'] = indoor_exposure
    debug['flash'] = flash; debug['BV'] = bv

    # --- base IO score ---
    if max(p_indoor, p_outdoor) < 0.55:
        in_vs_out += 0
    elif p_indoor > p_outdoor:
        in_vs_out += 12
    else:
        in_vs_out += 12
    in_vs_out = max(-15, min(20, in_vs_out))

    # --- day/night score ---
    strong_indoor  = (p_indoor  >= 0.7)
    strong_outdoor = (p_outdoor >= 0.7)
    if strong_indoor:
        if (local_is_day and p_day > p_night) or ((not local_is_day) and p_night > p_day):
            daynight += 3
        else:
            if (isinstance(flash, int) and flash != 0) or (isinstance(bv, (int,float)) and bv < 0) or low_light:
                daynight += 0
            else:
                daynight -= 2
    elif strong_outdoor:
        daynight += 10 if (p_day > p_night) == local_is_day else -9
    else:
        daynight += 4 if (p_day > p_night) == local_is_day else -3

    # --- geocode + nearby structure ---
    ge = reverse_geocode_struct(lat, lon, api_key)
    debug['geocode_location_type'] = ge['location_type']
    debug['geocode_address'] = ge['address']

    nearby_building = False
    if ge['loc'] is not None:
        # keep radius 20m; logic below will be stricter about using it
        nearby_building = places_establishment_within(ge['loc'][0], ge['loc'][1], api_key, radius_m=20)
    ge_loc_type   = (ge.get("location_type") or "")
    ge_is_precise = ge_loc_type in {"ROOFTOP", "RANGE_INTERPOLATED"}
    near_structure_ok = nearby_building or ge_is_precise
    debug['nearby_building_20m'] = nearby_building
    debug['ge_is_precise'] = ge_is_precise
    debug['near_structure_ok'] = near_structure_ok

    # --- sky fraction (top half) ---
    try:
        sf_top = sky_fraction_top(np_img, top_frac=0.5)
    except Exception:
        sf_top = 0.0
    sky_heavy = sf_top >= 0.20
    debug['sky_frac_top'] = sf_top
    debug['sky_heavy'] = sky_heavy

    # --- bring in EXIF PoP score to intensify penalty if needed ---
    pop_meta_here = pop_score_from_exif(exif_md) if isinstance(exif_md, (dict, str)) else {"score":0.0,"decision":"unlikely_pop"}
    debug['pop_exif_score'] = pop_meta_here.get("score", 0.0)
    debug['pop_exif_decision'] = pop_meta_here.get("decision", "unknown")
    pop_exif_strong = (debug['pop_exif_score'] >= 0.45) or (exif_flatness >= 0.60 and indoor_exposure == 1.0)
    debug['pop_exif_strong'] = pop_exif_strong

    # ---- Trusted outdoor guardrail (porches/overhangs) ----
    strong_outdoor_trust = (
        p_outdoor >= 0.95 and
        disp['real'] >= 0.30 and
        (not pop_exif_strong) and
        specular_ratio < 0.02
    )
    # If trusted outdoor and at least some sky, count it as "sky heavy"
    if strong_outdoor_trust and sf_top >= 0.12:
        sky_heavy = True
    debug['strong_outdoor_trust'] = strong_outdoor_trust
    debug['sky_heavy_after_trust'] = sky_heavy

    # Now that we have sky_heavy and pop_exif_strong, run displayed-media anti-cheat
    screen_signal = (disp['screen'] >= 0.65) or screen_like
    aux_cues_indoor = (specular_ratio > 0.03) or frame_present or context_indoor
    aux_cues_outdoor = ((not sky_heavy) or frame_present or context_indoor or (specular_ratio > 0.04))

    displayed_media_strong = False
    if p_indoor >= 0.7:
        if screen_signal and aux_cues_indoor:
            displayed_media_strong = True
        if (disp['window'] >= 0.70 and (frame_present or context_indoor) and (specular_ratio > 0.03)):
            displayed_media_strong = True
        # Guardrail: very strong indoor, no frame/context, and no EXIF PoP signal → do not flag
        if (p_indoor >= 0.90) and (not frame_present) and (not context_indoor) and (not pop_exif_strong):
            displayed_media_strong = False
    elif p_outdoor >= 0.7:
        # TIGHTER for outdoor: demand stronger cues
        if disp['screen'] >= 0.65 and (frame_present or context_indoor or specular_ratio > 0.035):
            displayed_media_strong = True
        elif screen_like and (frame_present or context_indoor) and (not sky_heavy) and pop_exif_strong:
            displayed_media_strong = True

    # --- NEW: decisive screen override (catches indoor PoP of outdoor scenes) ---
    # If the image content is overwhelmingly on a screen and there's no sky,
    # treat as displayed media regardless of IO classification.
    decisive_screen = (disp['screen'] >= 0.85) and (disp['real'] <= 0.15) and (sf_top < 0.05)
    debug['decisive_screen'] = decisive_screen
    if decisive_screen:
        displayed_media_strong = True

    debug['displayed_media_strong'] = displayed_media_strong

    if displayed_media_strong and not strong_outdoor_trust:
        # Heavier penalty if decisive screen; otherwise as before
        dm_pen = -22 if decisive_screen else -14
        if disp['screen'] >= 0.65:
            dm_pen -= 6
        if specular_ratio > 0.02:
            dm_pen -= 2
            cheat_reasons.append("glossy_specular")
        cheat_reasons.append(
            f"displayed_media(screen={disp['screen']:.2f}, window={disp['window']:.2f})"
        )
        # Cancel/limit environmental bonuses when it's displayed media
        daynight = min(daynight, 0)
        in_vs_out = max(-15, in_vs_out - 12)
        adjustments['cheat_penalty'] += dm_pen

    # --- anti-cheat (v3): screen / frame / no-sky near a precise structure ---
    cue_count = 0
    if screen_like:
        cue_count += 1
    if (context_indoor or frame_present):
        cue_count += 1
    if (not sky_heavy):
        cue_count += 1

    # Stricter again, but allow triggering without EXIF PoP when screen is decisive
    suspicious_outdoor_near_bldg = (
        (p_outdoor - p_indoor) >= 0.30 and
        nearby_building and
        (not sky_heavy) and
        (
            decisive_screen or
            (disp['screen'] >= 0.65) or
            (screen_like and (frame_present or context_indoor or specular_ratio > 0.03))
        ) and
        (not strong_outdoor_trust)
    )
    debug['suspicious_outdoor_near_bldg'] = suspicious_outdoor_near_bldg

    if suspicious_outdoor_near_bldg:
        base_pen = -18
        if screen_like or decisive_screen: cheat_reasons.append("screen_like_content")
        if context_indoor or frame_present: cheat_reasons.append("frame_or_indoor_borders")
        if not sky_heavy: cheat_reasons.append("no_real_sky")
        if pop_exif_strong:
            base_pen -= 6
            cheat_reasons.append("EXIF_flat+indoor_exposure_or_meta")
        if specular_ratio > 0.02:
            base_pen -= 2
            cheat_reasons.append("glossy_specular")
        # damp rewards and nudge IO down
        daynight = min(daynight, 0)
        in_vs_out = max(-15, in_vs_out - 8)
        adjustments['cheat_penalty'] += base_pen

    # optional: cap magnetometer if clearly indoor & near building
    if (p_indoor > 0.65 and nearby_building):
        adjustments['magnetometer_cap'] = 2

    # Final guardrail: don't let penalty nuke trusted-outdoor shots
    if strong_outdoor_trust:
        adjustments['cheat_penalty'] = min(adjustments['cheat_penalty'], -8)

    debug['cheat_penalty'] = adjustments['cheat_penalty']
    debug['cheat_reasons'] = cheat_reasons
    debug['in_vs_out_score'] = in_vs_out
    debug['daynight_score'] = daynight

    return in_vs_out, daynight, adjustments, debug


def pop_score_from_exif(exif_md_or_json):
    """
    Compute a Photo-of-Photo (PoP) score using ONLY metadata already parsed in memory.
    Accepts either:
      - dict: the value of data['imageMetadata']
      - str:  a JSON string containing the same structure

    Returns:
      {
        "score": float in [0,1],
        "decision": "likely_pop" | "borderline" | "unlikely_pop",
        "features": {
          "flatness": 0..1,
          "af_tight_center": 0/1,
          "hdr_small": 0/1,
          "flash_suspicious": 0/1,
          "indoor_pattern": 0/1,
          "orientation_incoherent": 0/0.5,
        },
        "debug": { ... raw values I used ... }
      }
    """
    # ---- Parse input (dict or JSON string) ----
    if isinstance(exif_md_or_json, str):
        try:
            md = json.loads(exif_md_or_json)
        except Exception:
            return {"score": 0.0, "decision": "unlikely_pop", "features": {}, "debug": {"error": "bad_json"}}
    elif isinstance(exif_md_or_json, dict):
        md = exif_md_or_json
    else:
        return {"score": 0.0, "decision": "unlikely_pop", "features": {}, "debug": {"error": "bad_input_type"}}

    # ---- Helpers for nested keys like "{Exif}.DateTimeOriginal" ----
    def get(d, path, default=None):
        cur = d
        for k in path.split('.'):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def g(path, default=None):
        # All keys in this payload live under imageMetadata
        # but sometimes the caller might pass only that subtree.
        root = md.get("imageMetadata", md)
        return get(root, path, default)

    # ---- Pull values (robustly) ----
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

    exposure_time = g("{Exif}.ExposureTime")  # seconds (float)
    fnum = g("{Exif}.FNumber")

    orient = g("{TIFF}.Orientation") or g("Orientation")
    gvec = g("{MakerApple}.8", [None, None, None])
    gx, gy, gz = (gvec + [None, None, None])[:3]

    # ---- Feature: flatness from gravity vector ----
    flatness = 0.0
    if isinstance(gy, (int, float)):
        # Phone nearly flat when |gy| small; map |gy| in [0,0.2] -> score in [1,0]
        if abs(gy) < 0.2:
            flatness = max(0.0, min(1.0, (0.2 - abs(gy)) / 0.2))
        else:
            flatness = 0.0

    # ---- Feature: AF area small & centered ----
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

    # ---- Feature: HDR on small subject ----
    hdr_small = 1.0 if (comp == 2 and af_tight_center == 1.0) else 0.0

    # ---- Feature: flash suspicious (flash in already bright scene) ----
    flash_suspicious = 0.0
    if isinstance(flash, int) and (flash != 0) and isinstance(bv, (int, float)):
        # Threshold 5.0 is a decent starting point; tune to your corpus.
        if bv > 5.0:
            flash_suspicious = 1.0

    # ---- Feature: indoor exposure pattern (weak prior) ----
    indoor_pattern = 0.0
    if isinstance(iso, (int, float)) and isinstance(exposure_time, (int, float)) and isinstance(fnum, (int, float)):
        if (150 <= iso <= 800) and (1/125 <= exposure_time <= 1/30) and (1.5 <= fnum <= 1.7):
            indoor_pattern = 1.0

    # ---- Feature: orientation coherence (tiny nudge) ----
    orientation_incoherent = 0.0
    if isinstance(orient, int) and isinstance(gy, (int, float)):
        # Very rough: portrait (6/8) expects |gy| high; landscape (1) expects |gy| smaller.
        if orient in (6, 8) and abs(gy) < 0.4:
            orientation_incoherent = 0.5
        if orient == 1 and abs(gy) > 0.9:
            orientation_incoherent = 0.5

    # ---- Weighted sum -> PoP score ----
    score = (
        0.55 * flatness +
        0.25 * af_tight_center +
        0.08 * hdr_small +
        0.06 * flash_suspicious +
        0.04 * indoor_pattern +
        0.02 * orientation_incoherent
    )
    score = max(0.0, min(1.0, float(score)))

    # ---- Decision thresholds (tune to your ROC) ----
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


def _convert_to_uT(x, y, z, units: str | None):
    """Convert magnetometer components to microtesla (µT) given units label."""
    if not units:
        return x, y, z
    u = units.strip().lower()
    # Known unit labels
    if u in {"ut", "µt", "microtesla", "micro-tesla"}:
        f = 1.0
    elif u in {"mt", "millitesla", "milli-tesla"}:
        f = 1000.0  # 1 mT = 1000 µT
    elif u in {"g", "gauss"}:
        f = 100.0   # 1 G = 100 µT
    elif u in {"mg", "milligauss", "milli-gauss"}:
        f = 0.1     # 1 mG = 0.1 µT
    else:
        f = 1.0
    return x * f, y * f, z * f

def calculate_magnetometer_score(
    data,
    exif_md=None,
    expected_mag=50.0,     # Earth's field magnitude target (µT)
    mag_tolerance=25.0,    # tolerance around expected magnitude (µT)
    heading_tolerance=30.0 # degrees tolerance for heading comparison
):
    # Pull heading (deg) and raw magnetometer payload
    heading_val = data.get('heading')
    raw_mag = data.get('magnetometerData')
    if raw_mag is None:
        print("Error: Missing magnetometer data")
        return -4

    # Parse magnetometer json if needed
    if isinstance(raw_mag, str):
        try:
            raw_mag = json.loads(raw_mag)
        except json.JSONDecodeError:
            print("Error: Invalid magnetometer JSON data")
            return -4

    # Extract components and units if provided
    try:
        x, y, z = float(raw_mag['x']), float(raw_mag['y']), float(raw_mag['z'])
    except (KeyError, ValueError, TypeError):
        print("Error: Magnetometer data missing or invalid x, y, or z components")
        return -4

    units = raw_mag.get('units')  # optional: "uT", "mT", "G", "mG"
    x, y, z = _convert_to_uT(x, y, z, units)

    # Heuristic auto-detection if units field is absent or wrong
    mag_tmp = math.sqrt(x*x + y*y + z*z)
    if units is None:
        if mag_tmp > 2000:    # likely mG provided
            x, y, z = x * 0.1, y * 0.1, z * 0.1
            units_used = "heuristic:mG->uT"
        elif mag_tmp < 1.0:   # likely mT provided
            x, y, z = x * 1000.0, y * 1000.0, z * 1000.0
            units_used = "heuristic:mT->uT"
        else:
            units_used = "uT(default)"
    else:
        units_used = f"{units}->uT"

    # Compute magnitude in µT
    mag = math.sqrt(x*x + y*y + z*z)
    print(f"Raw magnitude: {mag:.1f} µT ({units_used})")

    # Score magnitude on a [0..2]
    mag_factor = max(0.0, 1.0 - abs(mag - expected_mag) / mag_tolerance)
    mag_score = mag_factor * 2.0
    print(f"Mag factor: {mag_factor:.2f} → mag_score: {mag_score:.2f}")

    # Heading score only if we can reliably compare
    heading_score = 0.0
    if heading_val is not None:
        # Try to estimate whether device is approximately flat using EXIF gravity
        gy = None
        if isinstance(exif_md, dict):
            root = exif_md.get("imageMetadata", exif_md)
            gvec = root.get("{MakerApple}.8", [None, None, None])
            if isinstance(gvec, list) and len(gvec) >= 2:
                try:
                    gy = float(gvec[1])
                except (TypeError, ValueError):
                    gy = None

        # If the device is nearly flat (|gy| small), device XY ~ horizontal plane →
        # atan2(y, x) is a reasonable magnetic heading proxy.
        horizontal_ok = (gy is not None and abs(gy) < 0.35)  # ~20°-ish tilt tolerance

        if horizontal_ok:
            try:
                heading = float(heading_val)
                heading_mag = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
                diff = abs(heading_mag - heading)
                diff = min(diff, 360.0 - diff)
                print(f"Heading: {heading:.1f}°, Measured(flat): {heading_mag:.1f}°, Δ={diff:.1f}°")

                heading_factor = max(0.0, 1.0 - diff / heading_tolerance)
                heading_score = heading_factor * 2.0
                print(f"Heading factor: {heading_factor:.2f} → heading_score: {heading_score:.2f}")
            except (ValueError, TypeError):
                print("Warning: Invalid heading value in payload; skipping heading score")
                heading_score = 0.0
        else:
            print("Heading skipped (device not flat enough; need attitude to compare reliably)")

    # Total score [0..4], rounded once
    total = mag_score + heading_score
    magnetometer_score = int(round(total))
    print(f"Total score before rounding: {total:.2f} → {magnetometer_score}")
    return magnetometer_score

"""
def calculate_magnetometer_score(data):
    # Extract heading and magnetometer data
    heading = data.get('heading')
    magnetometerData = data.get('magnetometerData')
    
    if not heading or not magnetometerData:
        print("Error: Missing heading or magnetometer data")
        return -4

    # Parse magnetometer data
    if isinstance(magnetometerData, str):
        try:
            magnetometerData = json.loads(magnetometerData)
        except json.JSONDecodeError:
            print("Error: Invalid magnetometer JSON data")
            return -4

    # Extract x, y, z components
    try:
        x, y, z = magnetometerData['x'], magnetometerData['y'], magnetometerData['z']
    except KeyError:
        print("Error: Magnetometer data missing x, y, or z components")
        return -4

    # Calculate magnitude and score
    mag = (x*x + y*y + z*z)**0.5
    mag_score = round(2 * max(0, 1 - abs(mag - 45) / 20))  # Graduated scoring, rounded

    # Calculate heading and score
    try:
        heading_mag = (math.degrees(math.atan2(y, x)) + 360) % 360
        diff = abs(heading_mag - float(heading))
        diff = min(diff, 360 - diff)
        heading_score = round(2 * max(0, 1 - diff / 15))  # Graduated scoring, rounded
    except ValueError:
        print("Error: Invalid heading value")
        return -4

    # Total score
    magnetometer_score = round(mag_score + heading_score)  # Rounded total
    print(f"Magnitude: {mag:.2f} μT, Score: {mag_score}")
    print(f"Heading Diff: {diff:.2f}°, Score: {heading_score}")
    print(f"Total Magnetometer Score: {magnetometer_score}")
    
    return magnetometer_score
"""

def calculate_sha256(image_path):
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def insert_capture(capture_dict):
    # Connect to the database
    conn = sqlite3.connect('db/truths.db')
    cursor = conn.cursor()

    # Ensure new columns exist (idempotent)
    try:
        cursor.execute("PRAGMA table_info('captures')")
        cols = {row[1] for row in cursor.fetchall()}
        for col in ["baro_score", "magnetometer_score", "cheat_penalty"]:
            if col not in cols:
                cursor.execute(f"ALTER TABLE captures ADD COLUMN {col} TEXT")
        conn.commit()
    except Exception as e:
        print(f"[DB] schema check/alter failed: {e}")

    # Define the SQLite query to insert a record (now with extra columns)
    insert_query = '''
    INSERT INTO captures (
        id, userid, os, lat, lon, altitude, original_signature, watermarked_signature,
        relevant_obj_tags, device_time, server_time, t_score, os_score,
        in_vs_out_score, day_vs_night_score, altitude_score, device_score,
        barometerData, magnetometerData, heading, vscore,
        baro_score, magnetometer_score, cheat_penalty
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    # Convert all values to strings to match the TEXT type in SQLite
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

    # Execute the insert query
    cursor.execute(insert_query, capture_data)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def decode_jwt(token):
    # Decode the JWT without verifying the signature
    decoded = jwt.decode(token, options={"verify_signature": False})
    return decoded

def create_jwt(kid, team_id, private_key_path):
    # use seconds, not ms
    current_time = int(time.time())
    exp_time     = current_time + 3600         # one hour later

    # load your .p8 as before
    with open(private_key_path, 'r') as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read().encode(),
            password=None,
            backend=default_backend()
        )

    payload = {
        'iss': team_id,
        'iat': current_time,
        'exp': exp_time
    }

    # you don’t need to re-specify "alg" in headers; PyJWT will handle that
    token = jwt.encode(
        payload,
        private_key,
        algorithm='ES256',
        headers={'kid': kid}
    )

    return token

def verify_device_token(token, transaction_id, timestamp):
    # 1) Generate your Server-to-Apple JWT
    bearer = create_jwt(kid, team_id, private_key_path)

    # 2) Prepare request
    url = "https://api.development.devicecheck.apple.com/v1/validate_device_token"
    headers = {
        'Authorization': f'Bearer {bearer}',
        'Content-Type': 'application/json'
    }
    payload = {
        'device_token': token,
        'transaction_id': transaction_id,
        'timestamp': timestamp
    }

    # 3) Send to Apple
    response = requests.post(url, json=payload, headers=headers)

    # 4) If Apple gives 200 → token is valid.
    if response.status_code == 200:
        return { 'success': True }

    # 5) Otherwise try to parse Apple’s error payload
    try:
        print(f"Error Verifying Device. Status: {response.status_code} Response: {response.json()}")
        print("---")
        print(token)
        print("---")
        return response.json()
    except ValueError:
        # No JSON? Return status for logging
        return {
            'success': False,
            'status_code': response.status_code,
            'body': response.text,
            'token_with_issue' : token
        }


def reverse_geocode(lat, lng, api_key, timeout=5):
    """Returns formatted address or 'nowhere' on any failure."""
    url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?latlng={lat},{lng}&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[reverse_geocode] request/json error: {e}")
        return "nowhere"

    if data.get("status") == "OK" and data.get("results"):
        return data["results"][0].get("formatted_address", "nowhere")
    else:
        print(f"[reverse_geocode] API status not OK: {data.get('status')}")
        return "nowhere"

def get_elevation(lat, lng, api_key):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C{lng}&key={api_key}"
    response = requests.get(url)
    data = json.loads(response.text)

    if data["status"] == "OK":
        elevation = data["results"][0]["elevation"]
        return elevation
    else:
        print(data)
        return "Error: " + data["status"]

def get_local_pressures(lat, lon, timeout=5):
    """Fetch local mean sea-level pressure (pressure_msl) and surface pressure using Open-Meteo.
    Returns a dict with keys: 'pressure_msl' and 'surface_pressure' in hPa when available.
    """
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

def baro_altitude_from_pressure(pressure_hpa, sea_level_hpa):
    """Estimate altitude in meters using the barometric formula (ISA approximation).
    h = 44330 * (1 - (P/P0)^(1/5.255))
    pressure_hpa and sea_level_hpa are in hPa.
    """
    try:
        P = float(pressure_hpa)
        P0 = float(sea_level_hpa)
        if P <= 0 or P0 <= 0:
            return None
        return 44330.0 * (1.0 - (P / P0) ** (1.0 / 5.255))
    except Exception:
        return None

def get_altitude(lat, lon):
    api_url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lon}"
    response = requests.get(api_url)
    data = response.json()

    # Extract altitude from the response
    altitude = data['results'][0]['elevation']
    return altitude

def is_near_building_google(address, api_key, threshold=20, timeout=5):
    """
    Returns True if Places API finds an establishment within `threshold` meters;
    on any error or non-OK status, returns False.
    """
    # 1️⃣ Geocode the address
    geocode_url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?address={requests.utils.quote(address)}&key={api_key}"
    )
    try:
        ge_resp = requests.get(geocode_url, timeout=timeout)
        ge_resp.raise_for_status()
        ge_data = ge_resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[is_near_building_google] geocode error: {e}")
        return False

    if ge_data.get("status") != "OK" or not ge_data.get("results"):
        print(f"[is_near_building_google] geocode status: {ge_data.get('status')}")
        return False

    loc = ge_data["results"][0]["geometry"]["location"]
    center = Point(loc["lng"], loc["lat"])

    # 2️⃣ Nearby search for establishments
    places_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={loc['lat']},{loc['lng']}"
        f"&radius={threshold}&type=establishment&key={api_key}"
    )
    try:
        pl_resp = requests.get(places_url, timeout=timeout)
        pl_resp.raise_for_status()
        pl_data = pl_resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[is_near_building_google] places error: {e}")
        return False

    if pl_data.get("status") != "OK" or not pl_data.get("results"):
        print(f"[is_near_building_google] places status: {pl_data.get('status')}")
        return False

    # 3️⃣ Check distances
    for place in pl_data["results"]:
        ploc = place["geometry"]["location"]
        d_m = geodesic(
            (loc['lat'], loc['lng']), 
            (ploc['lat'], ploc['lng'])
        ).meters
        if d_m <= threshold:
            return True

    return False

"""
def is_near_building_google(address, api_key, threshold=20, timeout=5):
    
    #Returns True if Places API finds an establishment within `threshold` meters;
    #on any error or non-OK status, returns False.
    
    # 1️⃣ Geocode the address
    geocode_url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?address={requests.utils.quote(address)}&key={api_key}"
    )
    try:
        ge_resp = requests.get(geocode_url, timeout=timeout)
        ge_resp.raise_for_status()
        ge_data = ge_resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[is_near_building_google] geocode error: {e}")
        return False

    if ge_data.get("status") != "OK" or not ge_data.get("results"):
        print(f"[is_near_building_google] geocode status: {ge_data.get('status')}")
        return False

    loc = ge_data["results"][0]["geometry"]["location"]
    center = Point(loc["lng"], loc["lat"])

    # 2️⃣ Nearby search for establishments
    places_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={loc['lat']},{loc['lng']}"
        f"&radius={threshold}&type=establishment&key={api_key}"
    )
    try:
        pl_resp = requests.get(places_url, timeout=timeout)
        pl_resp.raise_for_status()
        pl_data = pl_resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[is_near_building_google] places error: {e}")
        return False

    if pl_data.get("status") != "OK" or not pl_data.get("results"):
        print(f"[is_near_building_google] places status: {pl_data.get('status')}")
        return False

    # 3️⃣ Check distances
    for place in pl_data["results"]:
        ploc = place["geometry"]["location"]
        if center.distance(Point(ploc["lng"], ploc["lat"])) <= (threshold / 1_000): 
            return True

    return False
"""

def is_near_building(address, threshold=20):
    # Step 1: Geocode the address
    geolocator = geopy.Nominatim(user_agent="Verisnap")
    location = geolocator.geocode(address)
    if not location:
        return False
    point = Point(location.longitude, location.latitude)

    # Step 2: Obtain building data (example with OpenStreetMap API)
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way
      (around:{threshold},{location.latitude},{location.longitude})
      ["building"];
    out geom;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Step 3: Check if any building is within the threshold distance
    for element in data['elements']:
        if 'geometry' in element:
            building_coords = [(node['lon'], node['lat']) for node in element['geometry']]
            building_polygon = Polygon(building_coords)
            if building_polygon.distance(point) <= threshold:
                return True
    return False

_tf = TimezoneFinder()

def is_daytime_at(lat, lon, iso_timestamp):
    """
    Returns True if the given ISO timestamp at (lat, lon) is between local sunrise and sunset.
    
    Requires: pip install python-dateutil astral timezonefinder pytz
    """
    # 1️⃣ Parse into an aware datetime (use UTC if no offset present)
    dt = parser.isoparse(iso_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # 2️⃣ Find the IANA zone for this lat/lon (fallback to UTC)
    tz_name = _tf.timezone_at(lat=lat, lng=lon) or "UTC"
    local_tz = pytz.timezone(tz_name)

    # 3️⃣ Convert timestamp into local time
    local_dt = dt.astimezone(local_tz)

    # 4️⃣ Compute sunrise & sunset for that date & location
    obs = Observer(latitude=lat, longitude=lon)
    s = sun(observer=obs, date=local_dt.date(), tzinfo=local_tz)

    # 5️⃣ Return whether between sunrise and sunset
    return s["sunrise"] <= local_dt <= s["sunset"]


def reverse_geocode_local(latitude, longitude):
    geolocator = Nominatim(user_agent="Verisnap")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    address = location.address if location else "nowhere"
    return address

#app = Flask(__name__)
#Compress(app)

def is_indoor_or_outdoor(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of an indoor scene", "a photo of an outdoor scene"]).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return "indoor" if probs[0, 0] > probs[0, 1] else "outdoor"
    
def is_day_or_night(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of a daylight scene", "a photo of a nighttime scene"]).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return "day" if probs[0, 0] > probs[0, 1] else "night"

def compare_version_strings(v1, v2):
    """
    Compare two version strings in the format of "major.minor.patch" and return:
    - 1 if version1 > version2
    - -1 if version1 < version2
    - 0 if version1 == version2
    """
    def parts(v): return [int(''.join(filter(str.isdigit, p)) or 0) for p in v.split('.')]
    a, b = parts(v1), parts(v2)
    for x, y in zip(a, b):
        if x != y: return 1 if x > y else -1
    return (1 if len(a) > len(b) and any(a[len(b):]) else
           -1 if len(b) > len(a) and any(b[len(a):]) else 0)

"""
def compare_version_strings(version1, version2):
    # Split the version strings into parts
    v1_parts = [int(part) for part in version1.split('.')]
    v2_parts = [int(part) for part in version2.split('.')]
    
    # Compare each part
    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 > v2:
            return 1
        elif v1 < v2:
            return -1
    
    # If all parts are equal so far, but one version has additional parts, compare those
    if len(v1_parts) > len(v2_parts):
        return 1 if any(part > 0 for part in v1_parts[len(v2_parts):]) else 0
    elif len(v1_parts) < len(v2_parts):
        return -1 if any(part > 0 for part in v2_parts[len(v1_parts):]) else 0

    # Versions are the same
    return 0
"""
"""
def compare_timestamp_with_current_utc(timestamp_str):
    # Parse the timestamp string into a datetime object
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
    
    # Assuming the given timestamp is in UTC, set its timezone to UTC
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    # Get the current time in UTC
    current_utc_time = datetime.now(timezone.utc)
    
    # Calculate the difference in milliseconds
    print(f"current app time={timestamp}")
    print(f"current server time={current_utc_time}")
    difference = abs((timestamp - current_utc_time).total_seconds() * 1000)  # Difference in milliseconds
    print(f"difference={difference}")
    
    # Compare the difference and return the score
    score = 51 if abs(difference) <= 30000 else 0
    return score
"""

def compare_timestamp_with_current_utc(timestamp_str):
    ts = parser.isoparse(timestamp_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    current_utc = datetime.now(timezone.utc)
    diff_ms = abs((ts - current_utc).total_seconds() * 1000.0)
    print(f"difference={diff_ms}")
    return 51 if diff_ms <= 30_000 else 0


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

"""
def add_watermark(input_image_path, output_image_path, watermark_text):
    photo = Image.open(input_image_path)
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
        # cases: image don't have getexif
        pass
    # make the image editable
    drawing = ImageDraw.Draw(photo)
    # Define the font and calculate text size
    #font = ImageFont.load_default()
    font_path = '/System/Library/Fonts/Arial.ttf'  # Replace with the correct path to Arial on your system
    #font_path = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
    font = ImageFont.truetype(font_path, 70)

    text_width = drawing.textlength(watermark_text, font=font)
    text_height = 14  # Approximation for single line

    # add watermark in the bottom right corner
    x = photo.width - (text_width) - 110
    y = photo.height - (text_height) - 120

    border_color = 'black'
    drawing.text((x-1, y-1), watermark_text, font=font, fill=border_color)
    drawing.text((x+1, y-1), watermark_text, font=font, fill=border_color)
    drawing.text((x-1, y+1), watermark_text, font=font, fill=border_color)
    drawing.text((x+1, y+1), watermark_text, font=font, fill=border_color)

    drawing.text((x, y), watermark_text, fill=(255,255,255), font=font)

    # by Verisnap

    font = ImageFont.truetype(font_path, 40)
    x1 = photo.width - (text_width) - 110
    y1 = photo.height - (text_height) - 120    
    x1 = x1 + 135
    y1 = y1 + 64    
    
    text_width = drawing.textlength("by Verisnap", font=font)
    text_height = 5  # Approximation for single line

    drawing.text((x1-1, y1-1), "by Verisnap", font=font, fill=border_color)
    drawing.text((x1+1, y1-1), "by Verisnap", font=font, fill=border_color)
    drawing.text((x1-1, y1+1), "by Verisnap", font=font, fill=border_color)
    drawing.text((x1+1, y1+1), "by Verisnap", font=font, fill=border_color)
    drawing.text((x1, y1), "by Verisnap", fill=(255,255,255), font=font)

    photo.save(output_image_path)
"""

def add_watermark(input_image_path, output_image_path, watermark_text):
    photo = Image.open(input_image_path)
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

    font_path = '/System/Library/Fonts/Arial.ttf'

    base_main_size = 70
    base_sub_size = 40
    font_main = ImageFont.truetype(font_path, int(base_main_size * 1.4))
    font_sub  = ImageFont.truetype(font_path, int(base_sub_size * 1.4))

    text_width = drawing.textlength(watermark_text, font=font_main)
    text_height = font_main.getbbox(watermark_text)[3]

    margin_x, margin_y = 110, 120
    x = photo.width - text_width - margin_x
    y = photo.height - text_height - margin_y

    # Determine local brightness under main label
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

    # Choose text/stroke for contrast
    if v_mean >= 0.6:
        fill_col = (0, 0, 0)
        stroke_col = (255, 255, 255)
    else:
        fill_col = (255, 255, 255)
        stroke_col = (0, 0, 0)

    stroke_w = max(2, font_main.size // 24)
    drawing.text((x, y), watermark_text, fill=fill_col, font=font_main,
                 stroke_width=stroke_w, stroke_fill=stroke_col)

    sub_text = "by Verisnap"
    text_width_sub = drawing.textlength(sub_text, font=font_sub)
    text_height_sub = font_sub.getbbox(sub_text)[3]

    x1 = x + int(135 * 1.4)
    y1 = y + int(64 * 1.4)

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

    photo.save(output_image_path)


def generate_thumbnail(source_path, thumb_path, size=(200,200)):
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    img = Image.open(source_path)
    img.thumbnail(size)
    img.save(thumb_path, format="JPEG")


def query_records(avghash):
    records_list = []  # List to store records as dictionaries
    try:
        # Connect to the database
        conn = sqlite3.connect('db/truths.db')
        cursor = conn.cursor()

        # Define the SQL query
        sql_query = "SELECT * FROM captures WHERE watermarked_signature = ?"

        # Execute the query
        cursor.execute(sql_query, (avghash,))
        
        # Fetch all matching records
        records = cursor.fetchall()

        # Process fetched records and convert to dictionaries
        for record in records:
            # Get column names
            columns = [col[0] for col in cursor.description]
            # Create a dictionary for the current record
            record_dict = dict(zip(columns, record))
            # Append the dictionary to the list
            records_list.append(record_dict)

    except sqlite3.Error as e:
        print("Error querying the database:", e)

    finally:
        # Close the connection
        if conn:
            conn.close()

    return records_list

def detect_picture_of_picture(path,
                              eps=0.05,
                              planar_thresh=0.15,      # flatness
                              var_thresh=0.06,         # low depth‐variance
                              edge_thresh=0.10,        # global edge density
                              border_thresh=0.02,      # 2% border edges
                              lbp_thresh=0.85,         # >85% uniform LBP
                              spec_thresh=0.95,        # >95% bright pixels = specular
                              gamut_thresh=0.50,       # avg channel-range <50% = narrow gamut
                              roi_frac=0.5):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # — Depth‐based features —
    inp = midas_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

    depth = (pred - pred.min()) / (pred.max() - pred.min())
    h, w = depth.shape
    rh, rw = int(h*roi_frac), int(w*roi_frac)
    y0, x0 = (h-rh)//2, (w-rw)//2
    roi = depth[y0:y0+rh, x0:x0+rw]

    planar_ratio = (np.abs(roi - np.median(roi)) < eps).sum() / roi.size
    variance     = float(np.var(roi))

    # — Global edge‐density —
    gray         = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges        = cv2.Canny(gray, 100, 200) > 0
    edge_density = edges.sum() / edges.size

    # — Border edge‐density —
    margin = int(0.05 * min(h, w))
    mask   = np.zeros_like(edges, bool)
    mask[:margin, :] = True; mask[-margin:, :] = True
    mask[:, :margin] = True; mask[:, -margin:] = True
    border_edge_density = edges[mask].sum() / mask.sum()

    # — LBP uniformity (texture) —
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    uniform = np.sum(lbp < 10)
    lbp_uniform_ratio = uniform / lbp.size

    # — Specular‐highlight ratio —
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2] / 255.0
    specular_ratio = np.mean(v > spec_thresh)

    # — Color gamut range —
    # compute per‐channel dynamic range
    r, g, b = arr[:,:,0]/255.0, arr[:,:,1]/255.0, arr[:,:,2]/255.0
    dr, dg, db = r.max()-r.min(), g.max()-g.min(), b.max()-b.min()
    gamut_range = np.mean([dr, dg, db])

    features = [[planar_ratio, variance, edge_density, border_edge_density]]
    pred = clf.predict(features)[0]       # 1 = p2p, 0 = real
    prob = clf.predict_proba(features)[0,1]  # confidence

    return bool(pred), planar_ratio, variance, edge_density, border_edge_density, prob

BYTES_PER_GB = 1_000_000_000  # decimal GB; change to 1_073_741_824 for GiB

def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                fp = os.path.join(root, name)
                total += os.path.getsize(fp)
            except OSError:
                # file might have been removed between walk and getsize
                pass
    return total

def _folder_report(base: str, name: str):
    path = os.path.join(base, name)
    size = _dir_size_bytes(path)
    count = 0
    if os.path.isdir(path):
        for _, _, files in os.walk(path):
            count += len(files)
    return {
        "path": path,
        "files": count,
        "bytes": size,
        "gb": round(size / BYTES_PER_GB, 6)
    }

def pop_cheat_from_features(p, r, bd):
    """
    Converts PoP probability and a simple geometric cue into a penalty.
    Context/frame cues are handled inside decide_indoor_outdoor and should
    not be penalized here to avoid double counting.
    """
    penalty = 0
    reasons = []

    # PoP probability bands
    if p >= 0.85:
        penalty -= 40; reasons.append(f"high PoP probability ({p:.2f})")
    elif p >= 0.65:
        penalty -= 25; reasons.append(f"elevated PoP probability ({p:.2f})")
    elif p >= 0.50:
        penalty -= 12; reasons.append(f"borderline PoP probability ({p:.2f})")

    # Geometric reinforcement (planarity + border edges)
    if r >= 0.40 and bd >= 0.015:
        penalty -= 6; reasons.append(f"planar_ratio={r:.2f} & border_edges={bd:.3f}")

    # Keep the overall anti-cheat contribution bounded
    penalty = max(-60, penalty)
    return penalty, reasons


@app.route('/me/storage', methods=['GET'])
@require_auth
def my_storage_report():
    """
    GET /me/storage
    Optional query:
      - uid=<userid>   -> check usage for this user
      - include_tmp=1  -> include storage/<uid>/tmp in totals
    Auth: Firebase ID token in Authorization: Bearer <token>
    """

    uid = request.user.get("email")

    include_tmp = request.args.get("include_tmp") == "1"

    base = os.path.join(app.root_path, 'storage', uid)
    breakdown = {
        "truths": _folder_report(base, "truths"),
        "thumbs": _folder_report(base, "thumbs"),
        "meta":   _folder_report(base, "meta"),
    }
    if include_tmp:
        breakdown["tmp"] = _folder_report(base, "tmp")

    total_bytes = sum(part["bytes"] for part in breakdown.values())
    result = {
        "userid": uid,
        "as_of": datetime.utcnow().isoformat(timespec='seconds') + "Z",
        "units": {"bytes_per_gb": BYTES_PER_GB, "gb_label": "GB (decimal)"},
        "include_tmp": include_tmp,
        "total": {
            "bytes": total_bytes,
            "gb": round(total_bytes / BYTES_PER_GB, 2)
        },
        "breakdown": breakdown
    }
    return jsonify(result), 200


@app.route('/ping',methods=['GET'])
def ping():
    return jsonify({
            'success': 'true',
            'message': f'pong ...'
        }), 200

@app.route('/check', methods=['POST'])
@require_auth
def check():
    # Handle optional gzip encoding
    raw = request.data
    if 'gzip' in request.headers.get('Content-Encoding', ''):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return jsonify({'success': 'false', 'message': 'Invalid gzip payload'}), 400

    try:
        data = json.loads(raw)
    except Exception:
        return jsonify({'success': 'false', 'message': 'Invalid JSON payload'}), 400

    pic_base64 = data.get('image')
    if not pic_base64:
        return jsonify({'success': 'false', 'message': 'Missing image'}), 400

    # Support data URLs
    if pic_base64.startswith('data:'):
        try:
            pic_base64 = pic_base64.split(',', 1)[1]
        except Exception:
            return jsonify({'success': 'false', 'message': 'Invalid data URL image'}), 400

    try:
        pic_data = base64.b64decode(pic_base64)
    except Exception:
        return jsonify({'success': 'false', 'message': 'Invalid base64 image'}), 400

    # Compute hash in-memory (no disk IO)
    try:
        with Image.open(BytesIO(pic_data)) as im:
            im = im.convert('RGB')
            avghash = imagehash.average_hash(im)
    except Exception:
        return jsonify({'success': 'false', 'message': 'Unreadable image'}), 400

    records = query_records(str(avghash))
    if len(records) >= 1:
        rec = records[0]
        try:
            score_val = float(rec.get('vscore', '0'))
        except Exception:
            score_val = 0.0
        print(f"Score was: {score_val}")
        if score_val >= 75:
            final_path = os.path.join('./', f"storage/{rec['userid']}/truths/{rec['id']}.jpeg")
            try:
                with open(final_path, 'rb') as final_file:
                    photo = final_file.read()
                base64_photo = base64.b64encode(photo).decode('utf-8')
            except FileNotFoundError:
                base64_photo = None
            return jsonify({
                'success': 'true',
                'truth': base64_photo,
                'message': f"Truth verified by Verisnap. TruthScore: {score_val} out of 100"
            }), 200
        else:
            return jsonify({
                'success': 'false',
                'message': f"Unable to verify the truth. TruthScore of {score_val} too low."
            }), 200

    return jsonify({
        'success': 'false',
        'message': 'Unable to verify the truth. Could not match with any existing certified picture.'
    }), 200

@app.route('/thumbnail/<userid>/<image_id>.jpeg', methods=['GET'])
def serve_thumbnail(userid, image_id):
    # storage/<userid>/thumbs/<image_id>_thumb.jpeg
    directory = os.path.join(app.root_path, 'storage', userid, 'thumbs')
    filename  = f"{image_id}_thumb.jpeg"
    return send_from_directory(directory, filename, mimetype='image/jpeg')

@app.route('/thumbnails', methods=['GET'])
def list_thumbnails():
    userid   = request.args.get('userid')
    page     = max(int(request.args.get('page', 1)), 1)
    per_page = min(int(request.args.get('per_page', 20)), 100)
    offset   = (page - 1) * per_page

    conn   = sqlite3.connect('db/truths.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, server_time FROM captures "
        "WHERE userid = ? "
        "ORDER BY server_time DESC "
        "LIMIT ? OFFSET ?",
        (userid, per_page, offset)
    )
    rows = cursor.fetchall()
    conn.close()

    base_url = request.url_root.rstrip('/')
    thumbs = []
    for image_id, server_ts in rows:
        url = f"{base_url}/thumbnail/{userid}/{image_id}.jpeg"
        thumbs.append({
            'id':    image_id,
            'time':  server_ts,
            'thumb': url
        })

    return jsonify({
        'page':     page,
        'per_page': per_page,
        'items':    thumbs
    }), 200

@app.route('/image/<userid>/<image_id>.jpeg', methods=['GET'])
def serve_full_image(userid, image_id):
    directory = os.path.join(app.root_path, 'storage', userid, 'truths')
    filename = f"{image_id}.jpeg"
    return send_from_directory(directory, filename, mimetype='image/jpeg')

@app.route('/capture/<userid>/<image_id>', methods=['GET'])
def get_capture(userid, image_id):
    conn = sqlite3.connect('db/truths.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM captures WHERE userid = ? AND id = ?",
        (userid, image_id)
    )
    row = cursor.fetchone()
    if not row:
        return jsonify({'error': 'Not found'}), 404
    columns = [col[0] for col in cursor.description]
    data = dict(zip(columns, row))
    conn.close()
    return jsonify(data), 200

# Read the label for a given picid
@app.route('/label/<picid>', methods=['GET'])
def get_label(picid):
    dataset_path = 'test/picofpicdataset.csv'
    if not os.path.exists(dataset_path):
        return jsonify({'label': 0}), 200
    with open(dataset_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['picid'] == picid:
                return jsonify({'label': int(row['label'])}), 200
    return jsonify({'label': 0}), 200

@app.route('/label', methods=['POST'])
def set_label():
    data        = request.get_json()
    picid       = data['picid']
    label       = str(data['label'])
    dataset_dir = 'test'
    dataset_path= os.path.join(dataset_dir, 'picofpicdataset.csv')
    temp_path   = dataset_path + '.tmp'

    # 1) make sure the folder exists
    os.makedirs(dataset_dir, exist_ok=True)

    rows       = []
    updated    = False
    # 2) load existing rows (if any)
    if os.path.exists(dataset_path):
        with open(dataset_path, newline='') as f:
            reader     = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['picid'] == picid:
                    row['label'] = label
                    updated = True
                rows.append(row)
    else:
        # if CSV doesn't exist yet, define its columns
        fieldnames = ['picid','planar_ratio','variance','edge_density','border_edge_density','label']

    # 3) if we never found that picid, append a new blank-feature row with this label
    if not updated:
        newrow = {fn: '' for fn in fieldnames}
        newrow['picid'] = picid
        newrow['label'] = label
        rows.append(newrow)

    # 4) write out to a temp file, then replace
    with open(temp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, dataset_path)

    return jsonify({'success': True}), 200


@app.route('/upload', methods=['POST'])
@require_auth
def upload():
    # Check if the data is GZip compressed
    if 'gzip' in request.headers.get('Content-Encoding', ''):
        print("Header was there")
        #try:
        # Decompress the data
        data = gzip.decompress(request.data)
        print("Data was decompress")
        #except OSError as e:
        #return jsonify({'error': 'Error decompressing gzip data'}), 400
    else:
        print("Header was not there.")
        data = request.data

    # Decode bytes to string and load JSON into a dictionary
    #try:
    #print(data)
    #data = json.loads(data.decode('utf-8'))
    data = json.loads(data)
    #except json.JSONDecodeError as e:
    #return jsonify({'error': 'Error decoding JSON data'}), 400
    #data = request.json
    #data = gzip.decompress(data)
    current_utc_time = datetime.now(timezone.utc)
    formatted_time = current_utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    # Extract data from the JSON
    userid = data.get('userid')
    deviceid = data.get('devicetoken')
    #print(f"Device token: {deviceid}")
    pic_base64 = data.get('pic')
    image_id = data.get('id')
    heading = data.get('heading')
    exif = data.get('imageMetadata') 

    # PoP from EXIF scoring

    pop_meta = pop_score_from_exif(exif)
    print(pop_meta)
    
    # Extract barometer and magnetometer compute scoring

    magnetometer_score = calculate_magnetometer_score(data, exif_md=exif)

    
    barometerData = data.get('barometerData')
    if barometerData is None:
        print("Barometer: None in payload")
    else:
        print("Barometer: payload present")
    magnetometerData = data.get('magnetometerData')
    """
    print(f"Barometer: {barometerData}")
    print(f"Magnetometer: {magnetometerData}")

    # right after you pull it out of the payload:
    raw_mag = magnetometerData

    # if it’s a string, parse it; otherwise assume it’s already a dict
    if isinstance(raw_mag, str):
        magnetometerData = json.loads(raw_mag)
    else:
        magnetometerData = raw_mag

    # parse and score:
    x,y,z = magnetometerData['x'], magnetometerData['y'], magnetometerData['z']
    mag = (x*x + y*y + z*z)**0.5
    mag_score = 2 if 25 < mag < 65 else -2

    heading_mag = (math.degrees(math.atan2(y, x)) + 360) % 360
    diff = abs(heading_mag - float(heading))
    diff = min(diff, 360 - diff)
    heading_score = 2 if diff < 15 else -2

    magnetometer_score = mag_score + heading_score
    """
    
    # Extract the 'location' field from the data
    location_json_str = data.get('location')
    location_data = json.loads(location_json_str)
    t = location_data.get('timestamp')
    latitude = location_data.get('latitude')
    longitude = location_data.get('longitude')
    altitude = location_data.get('altitude')
    
    osversion = data.get('operatingSystem')
    print(f"Operating System: {osversion}")

    create_folder_if_not_exists(f"storage/{userid}")
    create_folder_if_not_exists(f"storage/{userid}/tmp")
    create_folder_if_not_exists(f"storage/{userid}/truths")
    create_folder_if_not_exists(f"storage/{userid}/meta")
    create_folder_if_not_exists(f"storage/{userid}/thumbs")

    if not pic_base64 or not image_id:
        return jsonify({'error': 'Invalid data received'}), 400

    # Decode the Base64 string and save the image as a JPEG file
    try:
        pic_data = base64.b64decode(pic_base64)
        temp_path = os.path.join('./', f'storage/{userid}/tmp/{image_id}_temp.jpeg')
        final_path = os.path.join('./', f'storage/{userid}/truths/{image_id}.jpeg')
        meta_path = os.path.join('./', f'storage/{userid}/meta/{image_id}.json')
        dataset_path = os.path.join('./', f'test/picofpicdataset.csv')

        with open(temp_path, 'wb') as temp_file:
            temp_file.write(pic_data)

        with open(meta_path, 'w') as meta_file:
            meta_file.write(json.dumps(data))

        """
        (
        is_p2p, r, v, ed, bd, lbp, spec, gamut
        ) = detect_picture_of_picture(temp_path)
        print(f"planar={r:.2f}, var={v:.6f}, edges={ed:.3f}, bord={bd:.3f}, "f"lbp={lbp:.3f}, spec={spec:.3f}, gamut={gamut:.3f}")
        if is_p2p:
            print("🚩 photo-of-photo detected")
        else:
            print("✅ looks like a real scene")

        # — append features to CSV for offline labeling —
        dataset_dir  = 'test'
        dataset_path = f'{dataset_dir}/picofpicdataset.csv'
        os.makedirs(dataset_dir, exist_ok=True)
        """

        is_p2p, r, v, ed, bd, p = detect_picture_of_picture(temp_path)
        print(f"p2p prob={p:.2f} (planar={r:.2f}, var={v:.6f}, edges={ed:.3f}, bord={bd:.3f})")

        if p > 0.5:
            print("🚩 photo-of-photo detected")
        else:
            print("✅ real scene")

        # write header if file is new
        write_header = not os.path.exists(dataset_path)
        with open(dataset_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    'picid',
                    'planar_ratio',
                    'variance',
                    'edge_density',
                    'border_edge_density',
                    'label'
                ])
            writer.writerow([
                image_id,
                f'{r:.6f}',
                f'{v:.6f}',
                f'{ed:.6f}',
                f'{bd:.6f}',
                0
            ])
        # Scoring time:
        deltaT = compare_timestamp_with_current_utc(t)
        if deltaT == 51:
            print("Within time threshold. +51")
        
        servertimestamp = int(time.time() * 1000)
        result = verify_device_token(deviceid, image_id, servertimestamp)
        if result.get('success'):
            device_score = 10
        else:
            device_score = -2
            print(result)
        devverified = result.get('success')
        print(f"\nDevice Verified: {devverified}\n")

        sha256_hash = calculate_sha256(temp_path)

        # Scoring OS security
        deltaOS = 5 if (compare_version_strings(osversion, "17.0.0") == 1) else 0

        # Indoor vs outdoor
        """
        in_vs_out = 0
        daynightscore = 0
        print("Entering indoor outdoor...") 
        result = is_indoor_or_outdoor(temp_path)
        print(f"The image is likely {result}.")  
        if (result=="indoor"):
            print("Geocoding in progress ...")
            #address = reverse_geocode_local(latitude, longitude)
            address = reverse_geocode(latitude, longitude, api_key)
            if (address!="nowhere"):
                isbuilding = is_near_building_google(address, api_key) #is_near_building(address, threshold=20)
                if (isbuilding):
                    in_vs_out = 15
                    print(f"Address is: {address} and is within 20 meters of a building or house +15.")
                    print(f"Adjusting magnetometer score to 2 as indoors are usually noisy.")
                    magnetometer_score = 2
            else:
                print("No address was retrieved -> {address} Failed to Geocode -15.")
                in_vs_out = -15
        elif (result=="outdoor"):
            day_or_night = is_day_or_night(temp_path)
            print(f"The image is been likely taken during the {day_or_night}.")
            if (day_or_night == "day"):
                day_at = "day" if is_daytime_at(latitude, longitude, t) else "night"
                if (day_at == "day"):
                    daynightscore = 10
                    in_vs_out = 15
                    print("Geocoder reports day time vscore +10")
                else:
                    # This logic is getting confuse when pic is taken from inside a car during daylight
                    daynightscore = -20
                    in_vs_out = 15
                    print("Geocoder reports night time vscore -20")
            elif (day_or_night == "night"):
                day_at = "day" if is_daytime_at(latitude, longitude, t) else "night"
                if (day_at == "night"):
                    daynightscore = 10
                    in_vs_out = 15
                    print("Geocoder reports night time vscore +10")
                else:
                    daynightscore = -20
                    in_vs_out = 15
                    print("Geocoder reports day time vscore -20")
        """
        print("Entering indoor/outdoor…")
        in_vs_out, daynightscore, adj, dbg = decide_indoor_outdoor(
            temp_path=temp_path,
            lat=latitude,
            lon=longitude,
            iso_ts=t,
            api_key=api_key,
            exif_md=exif,
            barometer=barometerData
        )
        print(dbg)

        # Anti-cheat from IO/geocode context (inside decide_)
        io_cheat = adj.get('cheat_penalty', 0)
        io_reasons = dbg.get('cheat_reasons', [])

        # Anti-cheat from PoP + simple geometry (outside decide_)
        pop_penalty, pop_reasons = pop_cheat_from_features(p, r, bd)

        # Combine (clip if you like)
        cheat_penalty = io_cheat + pop_penalty
        cheat_penalty = max(-60, cheat_penalty)

        reasons_list = io_reasons + pop_reasons
        print("Anti-cheat penalty:", cheat_penalty, "Reasons:", reasons_list)

        # If we got a suggested magnetometer cap, apply it softly
        if adj.get('magnetometer_cap') is not None:
            magnetometer_score = min(magnetometer_score, adj['magnetometer_cap'])

        altitudeAtLocation = get_elevation(latitude, longitude, api_key)
        print(f"Calculated altitude was: {altitudeAtLocation} while received altitude was {altitude}.")

        try:
            ref_alt = float(altitudeAtLocation)
            dev_alt = float(altitude)
            if not np.isfinite(ref_alt) or not np.isfinite(dev_alt):
                raise ValueError("non-finite")
            deltaAltitude = abs(ref_alt - dev_alt)
        except (TypeError, ValueError):
            deltaAltitude = float('inf')  # force a penalty if we can't trust the number

        # Optional: use barometer to estimate altitude via local sea-level pressure
        baro_alt = None
        baro_score = 0
        if barometerData is not None:
            try:
                raw_baro = barometerData
                if isinstance(raw_baro, str):
                    raw_baro = json.loads(raw_baro)
                # iOS CMAltimeter pressure is in kPa typically
                p_kpa = raw_baro.get('pressure')
                # Optional timestamp sanity check (expect near the capture time)
                baro_ts_ok = True
                try:
                    baro_ts_str = raw_baro.get('timestamp')
                    if baro_ts_str and isinstance(t, str):
                        baro_dt = parser.isoparse(baro_ts_str)
                        loc_dt = parser.isoparse(t)
                        if baro_dt.tzinfo is None:
                            baro_dt = baro_dt.replace(tzinfo=timezone.utc)
                        if loc_dt.tzinfo is None:
                            loc_dt = loc_dt.replace(tzinfo=timezone.utc)
                        ts_diff = abs((baro_dt - loc_dt).total_seconds())
                        if ts_diff > 300:  # >5 minutes difference
                            baro_ts_ok = False
                            print(f"[barometer] timestamp stale by {ts_diff:.1f}s → ignoring barometer for scoring")
                except Exception as e:
                    print(f"[barometer] timestamp parse error: {e}")
                    baro_ts_ok = True

                if isinstance(p_kpa, (int, float)) and baro_ts_ok:
                    p_hpa = float(p_kpa) * 10.0
                    meteo = get_local_pressures(latitude, longitude)
                    p0 = meteo.get('pressure_msl')
                    if not isinstance(p0, (int, float)):
                        # Fallback: estimate sea-level pressure from surface pressure and ref altitude
                        ps = meteo.get('surface_pressure')
                        if isinstance(ps, (int, float)) and isinstance(ref_alt, float) and np.isfinite(ref_alt):
                            # P0 = Ps / (1 - h/44330)^(5.255)
                            try:
                                p0 = ps / (1.0 - (ref_alt / 44330.0)) ** 5.255
                                print(f"Estimated P0 from surface_pressure: Ps={ps:.1f} hPa, h={ref_alt:.1f} m → P0={p0:.1f} hPa")
                            except Exception:
                                p0 = None
                    if isinstance(p0, (int, float)):
                        baro_alt = baro_altitude_from_pressure(p_hpa, p0)
                        if baro_alt is not None and np.isfinite(baro_alt):
                            print(f"Barometer-estimated altitude: {baro_alt:.2f} m (P={p_hpa:.1f} hPa, P0={p0:.1f} hPa)")
                            if abs(ref_alt - baro_alt) < 15.0:
                                baro_score = 2
                            elif abs(ref_alt - baro_alt) > 80.0:
                                baro_score = -2
            except Exception as e:
                print(f"[barometer] error parsing/using barometer data: {e}")

        alt_thresh = 30.0
        if dbg.get('p_indoor', 0) >= 0.7 and dbg.get('nearby_building_20m'):
            alt_thresh = 60.0
        if dbg.get('context_indoor_override') or dbg.get('frame_present'):
            alt_thresh = max(alt_thresh, 80.0)

        # If we got a barometric estimate, consider the best agreement to determine score
        if baro_alt is not None and np.isfinite(deltaAltitude):
            delta_for_score = min(deltaAltitude, abs(ref_alt - baro_alt))
        else:
            delta_for_score = deltaAltitude

        if delta_for_score < alt_thresh:
            deltaAltitudeScore = 5
            print(f"Altitude difference {delta_for_score:.2f} < {alt_thresh} → +5")
        else:
            deltaAltitudeScore = -5
            print(f"Altitude difference {delta_for_score:.2f} ≥ {alt_thresh} → -5")


        vscore = deltaT + deltaOS + in_vs_out + daynightscore + deltaAltitudeScore + device_score + magnetometer_score + baro_score + cheat_penalty

        # print a breakdown plus the total
        print("TruthScore breakdown:")
        print(f"  Time score         (deltaT):               {deltaT}")
        print(f"  OS version score   (deltaOS):              {deltaOS}")
        print(f"  Indoor/Outdoor     (in_vs_out):            {in_vs_out}")
        print(f"  Day/Night match    (daynightscore):        {daynightscore}")
        print(f"  Altitude diff      (deltaAltitudeScore):   {deltaAltitudeScore}")
        print(f"  Barometer check    (baro_score):           {baro_score}")
        print(f"  Device verify      (device_score):         {device_score}")
        print(f"  Magnetometer check (magnetometer_score):   {magnetometer_score}")
        print(f"  Anti-cheat         (cheat_penalty):   {cheat_penalty}  -> {', '.join(reasons_list) or 'none'}")
        print("-" * 50)
        print(f"Total TruthScore (vscore):                   {vscore}")
        
        # Add watermark
        add_watermark(temp_path, final_path, f"TruthScore: {vscore}%")

        # Thumbnail
        thumb_path = os.path.join(f"storage/{userid}/thumbs/{image_id}_thumb.jpeg")
        generate_thumbnail(final_path, thumb_path)

        # Remove the temp file
        os.remove(temp_path)
        
        with open(final_path, 'rb') as final_file:
            photo = final_file.read()
        # Correctly format the base64 string
        base64_photo = base64.b64encode(photo).decode('utf-8')

        # Example usage:
        
        image4signature = Image.open(final_path)
        avghash = imagehash.average_hash(image4signature)
        
        capture_data = {
            'barometerData': barometerData,
            'magnetometerData': magnetometerData,
            'heading': heading,
            'userid': userid,
            'id': image_id,
            'os': str(osversion),
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
            'baro_score': str(baro_score),
            'magnetometer_score': str(magnetometer_score),
            'cheat_penalty': str(cheat_penalty)
        }

        insert_capture(capture_data)
        
        return jsonify({
            'success': 'true',
            'truth': base64_photo,
            'message': f'Image saved as {image_id}.jpeg with watermark'
        }),200
        
        '''
        with open(final_path, 'rb') as final_file:
            photo = final_file.read()
    
        # Correctly format the base64 string
        base64_photo = base64.b64encode(photo).decode('utf-8')

        # Prepare the response data as a JSON string
        response_data = json.dumps({
            'success': 'true',
            'truth': base64_photo,
            'message': f'Image saved as {image_id}.jpeg with watermark'
        })

        # Gzip the response data
        gzipped_response_data = gzip.compress(response_data.encode('utf-8'))

        # Create a Flask response
        response = make_response(gzipped_response_data)
    
        # Set the content type to JSON and specify the content encoding as gzip
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Encoding'] = 'gzip'
        '''
        return response  
              
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': 'false','error': str(e)}), 500

if __name__ == '__main__':
    #app.run(host='10.108.0.2', port=9000, debug=True)
    app.run(host='0.0.0.0', port=9000, debug=True)
