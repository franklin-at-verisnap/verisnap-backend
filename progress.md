Date: 2025-08-31

Session progress and changes

- Watermark readability (add_watermark)
  - Implemented adaptive text color and stroke based on local image brightness under the label.
    - If the sampled area is bright (V >= 0.6 in HSV), render black text with a white stroke.
    - Otherwise, render white text with a black stroke.
  - Uses Pillow stroke_width/stroke_fill for a scalable outline (thicker on larger fonts) to ensure visibility on any background.
  - Applies the same logic to the "by Verisnap" sub-label.

- Magnetometer precision
  - Unified magnetometer units to microtesla (µT) with robust conversion (uT/µT, mT, G, mG) and heuristics when units are missing.
  - Heading comparison is now computed only if the device is approximately flat (gated by EXIF gravity vector {MakerApple}.8), preventing misleading heading penalties.
  - Logging shows raw magnitude in µT, unit conversion used, and heading comparison details when applicable.
  - upload() now calls calculate_magnetometer_score(data, exif_md=exif).

- Displayed-media anti-cheat (screen/frame/window)
  - Deferred gating until after sky and EXIF PoP are computed to avoid undefined state and to reduce false positives.
  - Indoor: require strong screen signal plus corroboration (gloss/frame/context). Window requires frame/context + gloss.
  - Outdoor: only consider screen penalty if sky is unconvincing (low sky fraction) or there are corroborating cues (frame/context/gloss). Convincing sky avoids penalty.
  - Guardrail: strong-indoor with no frame/context and no EXIF PoP signal will not be penalized as displayed media.
  - Outdoor-near-building suspicion now requires multiple cues (at least two among screen_like, frame/context, low sky) to trigger.
  - Reduced penalty magnitudes to avoid over-penalizing legitimate shots; added detailed cheat_reasons in debug.

- Barometer integration
  - Added get_local_pressures(lat, lon) using Open-Meteo to fetch pressure_msl and surface_pressure.
  - Compute barometric altitude via ISA formula using device pressure (kPa → hPa) and P0 = pressure_msl; fallback to estimate P0 from surface_pressure and reference altitude when needed.
  - Added baro_score to TruthScore: +2 if |ref_elevation − baro_alt| < 15 m; −2 if > 80 m; else 0.
  - Altitude scoring now uses the best agreement between device altitude and barometric estimate.
  - Added logging: barometer payload presence, estimated P0 when using fallback, and barometer-estimated altitude details.
  - Timestamp sanity check for barometerData.timestamp (ISO8601); ignore barometer if stale by > 5 minutes from capture timestamp.

- Database and API
  - Extended captures table with new columns (added idempotently on insert): baro_score, magnetometer_score, cheat_penalty.
  - insert_capture now stores these fields; /capture returns them automatically for new rows.
  - TruthScore breakdown now prints Barometer check (baro_score).

Notes / Next steps

- iOS camera UX (client-side): ensure default to the wide lens (1x), add pinch-to-zoom and lens toggles (0.5x/1x/2x/5x) to avoid forced long-distance shots, and enable continuous AF with tap-to-focus.
- Potential enhancements:
  - Consider requiring two corroborating cues for displayed-media even indoors if CLIP false positives persist.
  - Store barometer timestamp and unit metadata in DB for deeper analysis.
  - Add optional drop-shadow or dynamic placement for the watermark if further robustness is needed over highly textured regions.
