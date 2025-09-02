CREATE TABLE captures (
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
    heading TEXT
);

ALTER TABLE captures ADD COLUMN vscore TEXT;

CREATE INDEX IF NOT EXISTS idx_captures_server_time 
  ON captures (server_time DESC);

