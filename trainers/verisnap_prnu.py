
"""
verisnap_prnu.py
-----------------
A lightweight PRNU wrapper for Verisnap.

What it does
============
• Enroll a device fingerprint from a batch of "flat" images (ideally textureless).
• Score a query image against a saved fingerprint using NCC and PCE.
• Produce auxiliary features useful for PoP detection: tile NCC stats and border drop.

Dependencies
============
pip install numpy opencv-python scikit-image scipy pillow tqdm

CLI
===
# Enroll from a folder of jpegs, save to device_fp.npz
python verisnap_prnu.py enroll ./calib/*.jpg --out device_fp.npz

# Score a single image against the fingerprint
python verisnap_prnu.py score device_fp.npz ./query.jpg

# Get JSON features only (silences pretty print)
python verisnap_prnu.py score device_fp.npz ./query.jpg --json
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from scipy.signal import wiener
from scipy.fft import fft2, ifft2
from scipy.stats import entropy
from tqdm import tqdm

EPS = 1e-8

# -----------------------
# Image / residual utils
# -----------------------

def _imread_gray(path: str) -> np.ndarray:
    """Read image as float32 grayscale in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = img.astype(np.float32) / 255.0
    return img

def _to_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop to the same size (min H,W)."""
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    def crop(x):
        y0 = (x.shape[0] - H) // 2
        x0 = (x.shape[1] - W) // 2
        return x[y0:y0+H, x0:x0+W]
    return crop(a), crop(b)

def _zero_mean_total(x: np.ndarray) -> np.ndarray:
    """Zero-mean total (ZMT): remove row/col means (as in PRNU literature)."""
    x = x - x.mean(axis=0, keepdims=True)
    x = x - x.mean(axis=1, keepdims=True)
    x = x + x.mean()
    return x

def estimate_noise_residual(img: np.ndarray, use_wiener: bool = True) -> np.ndarray:
    """
    Estimate noise residual W = I - denoise(I).
    Uses wavelet denoising + optional small Wiener smoothing to suppress non-PRNU content.
    """
    den = denoise_wavelet(
        img,
        method="BayesShrink",
        mode="soft",
        rescale_sigma=True,
        channel_axis=None
    ).astype(np.float32)
    w = img - den
    if use_wiener:
        # Light smoothing to reduce outliers while preserving sensor pattern
        w = wiener(w, (3, 3))
        w = w.astype(np.float32)
    return _zero_mean_total(w)

# -----------------------
# Fingerprint enrollment
# -----------------------

@dataclass
class Fingerprint:
    K: np.ndarray  # the PRNU fingerprint (float32)
    H: int
    W: int
    meta: Dict

    def save(self, path: str):
        np.savez_compressed(path, K=self.K, H=self.H, W=self.W, meta=json.dumps(self.meta))

    @staticmethod
    def load(path: str) -> "Fingerprint":
        d = np.load(path, allow_pickle=True)
        K = d["K"].astype(np.float32)
        H = int(d["H"])
        W = int(d["W"])
        meta = json.loads(str(d["meta"]))
        return Fingerprint(K=K, H=H, W=W, meta=meta)

def enroll_fingerprint(image_paths: List[str]) -> Fingerprint:
    """
    Estimate fingerprint K using the standard aggregation:
        K = sum( I_i * W_i ) / ( sum( I_i^2 ) + EPS ),
    where W_i is the residual of image I_i.
    """
    acc_num = None
    acc_den = None
    used = 0
    sizes = []

    for p in tqdm(image_paths, desc="Enrolling PRNU"):
        img = _imread_gray(p)
        sizes.append(img.shape[::-1])  # (W,H)

        w = estimate_noise_residual(img)
        if acc_num is None:
            acc_num = np.zeros_like(img, dtype=np.float32)
            acc_den = np.zeros_like(img, dtype=np.float32)

        img, w = _to_same_size(img, w)
        acc_num += img * w
        acc_den += img * img
        used += 1

    if used == 0:
        raise ValueError("No valid images provided for enrollment.")

    K = acc_num / (acc_den + EPS)
    K = _zero_mean_total(K).astype(np.float32)

    H, W = K.shape
    meta = {
        "num_images": used,
        "sizes": sizes[:10] + (["..."] if len(sizes) > 10 else []),
        "note": "Use images with low texture and adequate exposure for best results."
    }
    return Fingerprint(K=K, H=H, W=W, meta=meta)

# -----------------------
# Matching / statistics
# -----------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two same-size arrays."""
    a, b = _to_same_size(a, b)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float((a * b).sum() / denom)

def _xcorr2_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Circular cross-correlation via FFT (same size arrays).
    Returns real-valued correlation surface.
    """
    a, b = _to_same_size(a, b)
    A = fft2(a)
    B = fft2(b)
    corr = ifft2(A * np.conj(B)).real
    return np.fft.fftshift(corr)  # center the peak

def _pce(corr: np.ndarray, exclude_radius: int = 4) -> Tuple[float, Tuple[int, int]]:
    """
    Peak-to-Correlation Energy (PCE).
    PCE = peak^2 / mean( corr^2 outside a small window around the peak ).
    """
    H, W = corr.shape
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = corr[peak_idx]

    yy, xx = np.ogrid[:H, :W]
    cy, cx = peak_idx
    mask = (yy - cy)**2 + (xx - cx)**2 <= (exclude_radius**2)

    noise = corr.copy()
    noise[mask] = 0.0
    denom = (noise**2).sum() / (np.size(noise) - mask.sum() + EPS)
    pce = float((peak_val**2) / (denom + EPS))
    return pce, (int(peak_idx[0]), int(peak_idx[1]))

def score_image_against_fp(
    image_path: str,
    fp: Fingerprint,
    tile: int = 128,
    stride: Optional[int] = None,
    border_ratio: float = 0.05
) -> Dict:
    """
    Compute PRNU scores and auxiliary features for an image.

    Returns dict with:
      prnu_ncc, prnu_pce, pce_peak, tile_mean, tile_std, tile_entropy, border_drop
    """
    img = _imread_gray(image_path)
    img_c, K = _to_same_size(img, fp.K)

    # Query residual
    W = estimate_noise_residual(img_c)
    W = _to_same_size(W, K)[0]

    # "Expected PRNU pattern" in the query: I * K
    S = img_c * K

    # Global metrics
    prnu_ncc = _ncc(W, S)
    corr = _xcorr2_fft(W, S)
    prnu_pce, pce_peak = _pce(corr)

    # Tile NCC map
    if stride is None:
        stride = tile  # non-overlapping by default
    H, Wd = W.shape
    vals = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, Wd - tile + 1, stride):
            Wy = W[y:y+tile, x:x+tile]
            Sy = S[y:y+tile, x:x+tile]
            vals.append(_ncc(Wy, Sy))
    vals = np.array(vals, dtype=np.float32)
    if vals.size == 0:
        # fallback single-tile (whole image)
        vals = np.array([prnu_ncc], dtype=np.float32)

    tile_mean = float(vals.mean())
    tile_std = float(vals.std())

    # Entropy of NCC distribution (coarse histogram 21 bins)
    hist, _ = np.histogram(vals, bins=21, range=(-1.0, 1.0), density=True)
    tile_entropy = float(entropy(hist + EPS))

    # Border drop: mean NCC inner vs outer band
    br = max(1, int(border_ratio * min(H, Wd)))
    inner = W[br:H-br, br:Wd-br]
    innerS = S[br:H-br, br:Wd-br]
    if inner.size == 0:
        border_drop = 0.0
    else:
        inner_ncc = _ncc(inner, innerS)
        # Outer band consists of 4 strips
        top = _ncc(W[:br, :], S[:br, :]) if br > 0 else 0.0
        bottom = _ncc(W[H-br:, :], S[H-br:, :]) if br > 0 else 0.0
        left = _ncc(W[:, :br], S[:, :br]) if br > 0 else 0.0
        right = _ncc(W[:, Wd-br:], S[:, Wd-br:]) if br > 0 else 0.0
        outer_ncc = np.mean([top, bottom, left, right])
        border_drop = float(inner_ncc - outer_ncc)

    return {
        "prnu_ncc": float(prnu_ncc),
        "prnu_pce": float(prnu_pce),
        "pce_peak": pce_peak,
        "tile_mean": tile_mean,
        "tile_std": tile_std,
        "tile_entropy": tile_entropy,
        "border_drop": border_drop
    }

# -----------------------
# CLI
# -----------------------

def _cmd_enroll(args):
    paths = []
    for p in args.images:
        if os.path.isdir(p):
            for fname in os.listdir(p):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    paths.append(os.path.join(p, fname))
        else:
            paths.append(p)
    paths = sorted(paths)
    if not paths:
        raise SystemExit("No images found. Provide files or a folder with images.")

    fp = enroll_fingerprint(paths)
    out = args.out or "device_fp.npz"
    fp.save(out)
    print(f"Saved fingerprint to: {out}")
    print(f"Meta: {json.dumps(fp.meta, indent=2)}")

def _cmd_score(args):
    fp = Fingerprint.load(args.fingerprint)
    feats = score_image_against_fp(args.image, fp, tile=args.tile, stride=args.stride, border_ratio=args.border_ratio)
    if args.json:
        print(json.dumps(feats))
    else:
        print("\nPRNU scores")
        print("============")
        for k, v in feats.items():
            print(f"{k:>14}: {v}")

def main():
    ap = argparse.ArgumentParser(description="Verisnap PRNU wrapper")
    sub = ap.add_subparsers()

    ap_enroll = sub.add_parser("enroll", help="Enroll a device fingerprint from images")
    ap_enroll.add_argument("images", nargs="+", help="Image files and/or folders")
    ap_enroll.add_argument("--out", default="device_fp.npz", help="Output fingerprint file")
    ap_enroll.set_defaults(func=_cmd_enroll)

    ap_score = sub.add_parser("score", help="Score an image against a fingerprint")
    ap_score.add_argument("fingerprint", help="Path to fingerprint .npz file")
    ap_score.add_argument("image", help="Query image")
    ap_score.add_argument("--tile", type=int, default=128, help="Tile size for local NCC")
    ap_score.add_argument("--stride", type=int, default=None, help="Stride for tile NCC (default=tile)")
    ap_score.add_argument("--border_ratio", type=float, default=0.05, help="Outer band width ratio for border_drop")
    ap_score.add_argument("--json", action="store_true", help="Print JSON only")
    ap_score.set_defaults(func=_cmd_score)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
