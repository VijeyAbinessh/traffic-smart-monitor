#!/usr/bin/env python3
# main_no_ui.py
# Smart Traffic Behavior Monitoring System (UI removed)
# Accepts --url or --video on command line and shows frames with cv.imshow

import os
import csv
import time
import math
import threading
import traceback
from collections import deque, defaultdict
import argparse

import cv2
import numpy as np
from ultralytics import YOLO

# removed tkinter imports (UI-less)
from PIL import Image, ImageDraw  # Pillow used only for some helpers retained
# joblib/torch for color classifier
import joblib
import torch
import torch.nn as nn

import folium
import json
import base64
import requests


# Ensure outputs folder exists and subfolders for violations (names per your request)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/speed", exist_ok=True)
os.makedirs("outputs/wrongway", exist_ok=True)
os.makedirs("outputs/occupancy", exist_ok=True)
os.makedirs("violations", exist_ok=True)
os.makedirs("map_snapshots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------- globals for top-level payload fields ----------
ZONE_ID = None        # set to an int like 78 if desired via args/env
LOCATION_ID = None    # set to an int like 567 if desired via args/env
FRAME_PAYLOAD_TYPE = "traffic_frame_analysis"

# ROI storage: map video_path -> (x1,y1,x2,y2) in frame coords
roi_boxes = {}
# Polygon ROI storage: map video_path -> list of (x,y) points (frame coords)
roi_polygons = {}
# UI drawing temporary state per panel (map path -> dict)
_roi_draw_state = {}

# ---------- helpers ----------
def rect_intersect_area(r1, r2):
    """Return True if rect r1 (x1,y1,x2,y2) and r2 intersect."""
    ax1,ay1,ax2,ay2 = r1
    bx1,by1,bx2,by2 = r2
    if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
        return False
    return True

def _normalize_subtype(subtype):
    if not subtype:
        return "unknown"
    s = str(subtype).strip().lower()
    if "speed" in s:
        return "speed"
    if ("wrong" in s and "way" in s) or s in ("wrongway", "wrong_way"):
        return "wrongway"
    if "heavy" in s and "traffic" in s:
        return "occupancy"
    if "occup" in s or "density" in s:
        return "occupancy"
    if s == "vehicle_type":
        return "vehicle_type"
    import re
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'[^a-z0-9_]+', '', s)
    s = re.sub(r'_{2,}', '_', s)
    return s or "unknown"

def point_in_polygon(x, y, poly):
    if not poly or len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        if (min(yi, yj) <= y <= max(yi, yj)) and (min(xi, xj) <= x <= max(xi, xj)):
            dx = xj - xi
            dy = yj - yi
            if abs(dx) < 1e-6 and abs(x - xi) < 1e-6:
                return True
            if abs(dx) > 1e-6:
                t = (x - xi) / dx
                y_on_line = yi + t * dy
                if abs(y_on_line - y) < 1e-6:
                    return True
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
    return inside

# Emit a payload and detection dict (does NOT write per-detection files)
def emit_alert_json_single_detection_snake(frame, bbox, violation_subtype, reference,
                                           speed=None, color=None, details=None, frame_location=None,
                                           reader_id=None, vehicle_type=None, frame_id=None,
                                           zone_id=None, location_id=None, analysis_type=None):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        tsfile = time.strftime("%Y%m%d_%H%M%S")

        if frame is not None and hasattr(frame, "shape"):
            frame_h = int(frame.shape[0]); frame_w = int(frame.shape[1])
        else:
            frame_h = 0; frame_w = 0

        reader_id = str(reader_id or os.environ.get("READER_ID", "reader_123"))
        subtype_norm = _normalize_subtype(violation_subtype) if violation_subtype else None

        coords = [0,0,0,0]
        try:
            coords = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
        except Exception:
            try:
                coords = [int(x) for x in bbox]
                if len(coords) < 4:
                    coords = (coords + [0,0,0,0])[:4]
            except Exception:
                coords = [0,0,0,0]
        x1,y1,x2,y2 = coords

        try:
            value_val = 0.0 if speed is None else float(speed)
        except:
            value_val = 0.0

        vtype_str = "unknown"
        try:
            vtype_str = str(vehicle_type).strip() if vehicle_type is not None else "unknown"
        except:
            vtype_str = "unknown"
        if not vtype_str:
            vtype_str = "unknown"

        color_str = str(color).strip() if color is not None else ""

        co_or_coords = [
            {"x": int(x1), "y": int(y1)},
            {"x": int(x2), "y": int(y1)},
            {"x": int(x2), "y": int(y2)},
            {"x": int(x1), "y": int(y2)}
        ]

        is_violation = (subtype_norm in ("speed", "wrongway", "occupancy"))
        if is_violation:
            detect = {
                "subtype": subtype_norm,
                "reference": str(reference) if reference is not None else None,
                "vehicle": vtype_str,
                "success": False,
                "value": float(value_val),
                "color": color_str.lower() if isinstance(color_str, str) else color_str,
                "imageLocation": None,
                "coOrDinates": co_or_coords,
                "timestamp": ts,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "violationType": subtype_norm
            }
        else:
            detect = {
                "subtype": "vehicle_type",
                "reference": str(reference) if reference is not None else None,
                "vehicle": vtype_str,
                "success": True,
                "value": float(value_val),
                "color": color_str.lower() if isinstance(color_str, str) else color_str,
                "imageLocation": None,
                "coOrDinates": co_or_coords,
                "timestamp": ts,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "AttributeDto": {"Type": vtype_str, "Colour": color_str}
            }

        if frame_id:
            frame_id_use = frame_id
        else:
            frame_id_use = (os.path.basename(frame_location).replace(" ", "_") + f"_{tsfile}")

        payload = {
            "type": analysis_type if analysis_type else FRAME_PAYLOAD_TYPE,
            "zoneId": None,
            "LocationId": None,
            "readerId": reader_id,
            "frameId": frame_id_use,
            "frameLocation": frame_location or "",
            "timestamp": ts,
            "frameWidth": int(frame_w),
            "frameHeight": int(frame_h),
            "detections": [detect]
        }

        zid = zone_id if zone_id is not None else globals().get("ZONE_ID", None)
        lid = location_id if location_id is not None else globals().get("LOCATION_ID", None)

        payload["zoneId"] = (int(zid) if (zid is not None and isinstance(zid, (int,str)) and str(zid).isdigit()) else (zid if zid is not None else None))
        payload["LocationId"] = (int(lid) if (lid is not None and isinstance(lid, (int,str)) and str(lid).isdigit()) else (lid if lid is not None else None))

        return payload, detect

    except Exception as e:
        print("[WARN] emit_alert_json failed:", e)
        return None, None

# ---------- Per-frame JSON writer (merge all detections into a single frame file) ----------
frame_json_lock = threading.Lock()

def _frame_filename_for_id(frame_id):
    safe = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in frame_id)
    return os.path.join("outputs", f"frame_{safe}.json")

def _detection_same(d1, d2):
    try:
        s1 = d1.get("subtype"); s2 = d2.get("subtype")
        if s1 is not None and s2 is not None and str(s1) != str(s2):
            return False

        r1 = d1.get("reference"); r2 = d2.get("reference")
        if r1 and r2 and str(r1) == str(r2):
            return True

        c1 = d1.get("coOrDinates"); c2 = d2.get("coOrDinates")
        if c1 is not None and c2 is not None and c1 == c2:
            return True

        b1 = d1.get("bbox"); b2 = d2.get("bbox")
        if b1 is not None and b2 is not None and b1 == b2:
            return True
    except Exception:
        pass
    return False

def write_frame_json(payload, detect):
    if not payload or not detect:
        return False
    frame_id = payload.get("frameId")
    if not frame_id:
        return False
    path = _frame_filename_for_id(frame_id)
    with frame_json_lock:
        existing = None
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = None
        if not existing or not isinstance(existing, dict):
            frame_entry = {
                "type": payload.get("type", FRAME_PAYLOAD_TYPE),
                "frameId": frame_id,
                "zoneId": payload.get("zoneId"),
                "LocationId": payload.get("LocationId"),
                "readerId": payload.get("readerId"),
                "frameLocation": payload.get("frameLocation", ""),
                "timestamp": payload.get("timestamp"),
                "frameWidth": payload.get("frameWidth", 0),
                "frameHeight": payload.get("frameHeight", 0),
                "detections": []
            }
            if "zoneId" in payload:
                frame_entry["zoneId"] = payload["zoneId"]
            if "LocationId" in payload:
                frame_entry["LocationId"] = payload["LocationId"]

            frame_entry["detections"].append(detect)
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump(frame_entry, f, indent=2)
                os.replace(path + ".tmp", path)
                print(f"[FRAME] Created {path} with 1 detection")
                return True
            except Exception as e:
                print("[WARN] write_frame_json create failed:", e)
                try:
                    if os.path.exists(path + ".tmp"):
                        os.remove(path + ".tmp")
                except:
                    pass
                return False
        else:
            if "type" not in existing and "type" in payload:
                existing["type"] = payload.get("type")
            if "zoneId" not in existing and "zoneId" in payload:
                existing["zoneId"] = payload.get("zoneId")
            if "LocationId" not in existing and "LocationId" in payload:
                existing["LocationId"] = payload.get("LocationId")

            detections = existing.setdefault("detections", [])
            is_dup = False
            for ex in detections:
                if _detection_same(ex, detect):
                    is_dup = True
                    break
            if not is_dup:
                detections.append(detect)
                try:
                    with open(path + ".tmp", "w") as f:
                        json.dump(existing, f, indent=2)
                    os.replace(path + ".tmp", path)
                    print(f"[FRAME] Appended detection to {path} (now {len(detections)} detections)")
                    return True
                except Exception as e:
                    print("[WARN] write_frame_json append failed:", e)
                    try:
                        if os.path.exists(path + ".tmp"):
                            os.remove(path + ".tmp")
                    except:
                        pass
                    return False
            else:
                return True

# ---------- End per-frame JSON helpers ----------

# Global map object
traffic_map = folium.Map(location=[12.91, 80.23], zoom_start=13)
video_location_default = (12.91, 80.23)

selected_videos = []
video_locations = {}
map_lock = threading.Lock()
video_ui = {}
sequential_id_map = {}
sequential_next = {}

def get_display_id(video_path, orig_tid):
    try:
        key = os.path.normpath(video_path)
    except:
        key = str(video_path)
    if key not in sequential_id_map:
        sequential_id_map[key] = {}
        sequential_next[key] = 1
    mapping = sequential_id_map[key]
    if orig_tid in mapping:
        return mapping[orig_tid]
    seq = sequential_next[key]
    mapping[orig_tid] = seq
    sequential_next[key] = seq + 1
    return seq

PANEL_W = 560; PANEL_H = 420

def plot_event_on_map(location, event_type, obj_id, snapshot_path=None):
    try:
        server = globals().get('MAP_SERVER_URL', 'http://127.0.0.1:5000')
        api_key = globals().get('MAP_API_KEY', '')
        payload = {
            'lat': float(location[0]) if location else None,
            'lon': float(location[1]) if location else None,
            'event_type': str(event_type),
            'obj_id': str(obj_id),
            'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source': ''
        }
        if snapshot_path and os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, 'rb') as f:
                    b = f.read()
                    b64 = base64.b64encode(b).decode('ascii')
                    payload['image'] = 'data:image/jpeg;base64,' + b64
            except Exception:
                pass
        if server:
            try:
                url = server.rstrip('/') + '/api/event'
                headers = {'Content-Type': 'application/json'}
                if api_key:
                    headers['X-API-Key'] = api_key
                requests.post(url, json=payload, headers=headers, timeout=5)
                print(f"[MAP] Posted event to map server: {url}")
                return
            except Exception as e:
                print(f"[WARN] Posting to map server failed: {e}")
    except Exception as e:
        print('[WARN] plot_event_on_map building payload failed:', e)
    try:
        html = f"<b>{event_type}</b><br>ID: {obj_id}<br>{time.strftime('%Y-%m-%d %H:%M:%S')}"
        if snapshot_path:
            html += f"<br><img src='{snapshot_path}' width='200'>"
        popup = folium.Popup(html, max_width=300)
        with map_lock:
            folium.Marker(
                location=location,
                popup=popup,
                icon=folium.Icon(color='red' if "Warning" in event_type else 'blue', icon="car")
            ).add_to(traffic_map)
            traffic_map.save("traffic_events.html")
        print(f"[MAP] {event_type} appended at {location} with snapshot {snapshot_path}")
    except Exception as e:
        print(f"[WARN] map update failed: {e}")

class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,9)
        )
    def forward(self,x):
        return self.net(x)

# load pre-trained classifier if available
color_model = ColorClassifier()
try:
    path_weights = r"D:\yash\abcd\traffic analysis bengalore demo\vehicle_detection\color_classifier.pth"
    if os.path.exists(path_weights):
        color_model.load_state_dict(torch.load(path_weights, map_location="cpu"))
        color_model.eval()
    else:
        color_model = None
        print("[WARN] color classifier not found; color detection disabled.")
except Exception as e:
    color_model = None
    print("[WARN] color classifier load error:", e)

label_encoder = None
try:
    path_le = r"D:\yash\abcd\traffic analysis bengalore demo\vehicle_detection\label_encoder.pkl"
    if os.path.exists(path_le):
        label_encoder = joblib.load(path_le)
    else:
        label_encoder = None
except Exception as e:
    label_encoder = None
    print("[WARN] label encoder load error:", e)

try:
    from sort import Sort
except Exception as e:
    Sort = None
    print("[WARN] sort.py import failed:", e)

try:
    import joblib
    from sklearn.cluster import KMeans
except Exception as e:
    joblib = None
    KMeans = None
    print("[WARN] joblib/sklearn not fully available:", e)

def cuda_setup():
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except:
                gpu_name = "GPU"
            print(f"[INFO] CUDA available. Using device: cuda ({gpu_name})")
            return "cuda"
    except Exception:
        pass
    print("[INFO] CUDA not available. Using CPU.")
    return "cpu"

DEVICE = cuda_setup()

shared_model = None
model_lock = threading.Lock()

INFERENCE_CONCURRENCY = 1
inference_semaphore = threading.Semaphore(INFERENCE_CONCURRENCY)

def get_shared_model():
    global shared_model
    if shared_model is None:
        with model_lock:
            if shared_model is None:
                try:
                    print("[INFO] Loading shared YOLO model...")
                    m = YOLO(MODEL_WEIGHTS)
                    try:
                        m.to(DEVICE)
                    except Exception:
                        pass
                    shared_model = m
                    print("[INFO] Shared YOLO model loaded.")
                except Exception as e:
                    print(f"[ERROR] Failed to load shared model: {e}")
                    shared_model = None
    return shared_model

# --------------------------- CONFIG / GLOBALS --------------------------- #
MODEL_WEIGHTS = "yolo12x.pt"   # change to your weights if needed
DET_CONF = 0.15
LABEL_IOU_THRESH = 0.05

MAP_SERVER_URL = os.environ.get('MAP_SERVER_URL', 'http://127.0.0.1:5000')
MAP_API_KEY = os.environ.get('MAP_API_KEY', '')

SORT_MAX_AGE = 50
SORT_MIN_HITS = 2
SORT_IOU_THRESHOLD = 0.10

frozen_speed = {}
PIXEL_TO_METER = 0.05
CALIBRATION_FACTOR = 3.0
SPEED_LIMIT = 60

STOP_SPEED_THRESHOLD = 5

VIOLATIONS_DIR = "violations"
violation_timers = {}
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

FRAME_FAILURE_RETRY_LIMIT = 5
AUTO_REOPEN_ON_FAIL = True

video_path = None
running = False

prev_positions = {}
current_speed = {}
direction_history = {}
zigzag_events = {}
stop_events = {}

SEEN_FRAMES_MIN = 3
COUNT_COOLDOWN_SEC = 5.0
seen_frames = {}
last_seen_ts = {}
counted_ids = set()

y_history = defaultdict(list)
wrongway_logged_ids = set()
vehicle_count_set = set()

car_count = bus_count = truck_count = 0
motorcycle_count = 0

lanes_user_count = 1
expected_direction_global = "in"

# --- lightweight shims to replace Tk variables and UI objects (no-UI mode) ---
class SimpleVar:
    def __init__(self, v=None):
        self.v = v
    def get(self):
        return self.v
    def set(self, x):
        self.v = x

# objects that code expects to exist; set to None or SimpleVar where needed
videos_listbox = None
file_label = None
videos_container = None
video_label = None

lanes_entry = SimpleVar("3")
direction_var = SimpleVar("in")
speed_limit_var = SimpleVar(60)
zone_id_var = SimpleVar("")
location_id_var = SimpleVar("")

vehicle_count_var = SimpleVar(0)
car_count_var = SimpleVar(0)
bus_count_var = SimpleVar(0)
truck_count_var = SimpleVar(0)
motorcycle_count_var = SimpleVar(0)

# UI tree placeholders not used in headless
log_list = None
violation_list = None
wrongway_list = None

# color classifier fallback paths and functions (unchanged)
COLOR_MODEL_PATH = "hsv_color_model_fixed5.pkl"
SHRINK_RATIO = 0.15
SAT_THRESHOLD = 20
K_CLUSTERS = 3

clf = None
if joblib is not None:
    try:
        if os.path.exists(COLOR_MODEL_PATH):
            clf = joblib.load(COLOR_MODEL_PATH)
            print(f"[INFO] Loaded color classifier from {COLOR_MODEL_PATH}")
        else:
            clf = None
            print(f"[WARN] Color classifier file not found at {COLOR_MODEL_PATH}. Color detection will return 'unknown'.")
    except Exception as e:
        clf = None
        print("[WARN] Failed to load color classifier:", e)

def crop_inner_box(x1,y1,x2,y2, shrink=SHRINK_RATIO):
    w = max(1, x2-x1); h = max(1, y2-y1)
    dx = int(w*shrink); dy = int(h*shrink)
    return x1+dx, y1+dy, x2-dx, y2-dy

def color_distribution_hsl(bgr_img, k=K_CLUSTERS):
    if bgr_img is None or bgr_img.size == 0 or clf is None or KMeans is None:
        return []
    try:
        roi = cv2.resize(bgr_img, (96,96))
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        mask = hls[:,:,2] > SAT_THRESHOLD
        if np.sum(mask) == 0:
            return []
        pixels = hls[mask].reshape(-1,3).astype(np.float32)
        n_clusters = min(k, len(pixels))
        if n_clusters <= 0:
            return []
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        total = float(np.sum(counts))
        if total <= 0: return []
        raw_results = []
        for i in range(len(counts)):
            H,L,S = kmeans.cluster_centers_[i]
            try:
                pred = clf.predict([[H, S, L]])[0]
            except Exception:
                pred = "unknown"
            pct = 100.0 * counts[i] / total
            raw_results.append((pred, pct))
        merged = {}
        for label, pct in raw_results:
            merged[label] = merged.get(label, 0.0) + pct
        results = sorted(merged.items(), key=lambda x: -x[1])
        return results
    except Exception as e:
        print("[WARN] color_distribution_hsl failed:", e)
        return []

def detect_roi_color_label(roi):
    if roi is None or roi.size == 0:
        return "unknown"
    if color_model is None or label_encoder is None:
        return "unknown"
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1]) * 100.0/255.0
        v_mean = np.mean(hsv[:,:,2]) * 100.0/255.0
        sample = np.array([[h_mean, s_mean, v_mean]], dtype=np.float32)
        sample_tensor = torch.tensor(sample)
        with torch.no_grad():
            outputs = color_model(sample_tensor)
            _, predicted = torch.max(outputs, 1)
            color_name = label_encoder.inverse_transform(predicted.numpy())
        return str(color_name[0])
    except Exception as e:
        print("[WARN] detect_roi_color_label failed:", e)
        return "unknown"

# --------------------------- LOGGING & SNAPSHOT (with color + JSON) ---------------------------
def save_violation_snapshot(frame, box, tid, violation_type, color_label=None, video_path=None, vehicle_type=None, frame_id=None):
    subtype_normal = _normalize_subtype(violation_type)
    outfolder = os.path.join("outputs", subtype_normal)
    try:
        os.makedirs(outfolder, exist_ok=True)
    except:
        pass

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    if color_label is None:
        try:
            cx1, cy1, cx2, cy2 = crop_inner_box(x1, y1, x2, y2)
            cx1 = max(0, min(cx1, w-1)); cx2 = max(0, min(cx2, w-1))
            cy1 = max(0, min(cy1, h-1)); cy2 = max(0, min(cy2, h-1))
            roi = frame[cy1:cy2, cx1:cx2]
            color_label = detect_roi_color_label(roi)
        except Exception as e:
            print("[WARN] color compute failed in save_violation_snapshot:", e)
            color_label = "unknown"

    ts = time.strftime('%Y%m%d_%H%M%S')
    safe_label = str(color_label).replace(" ", "_")
    safe_video = os.path.basename(video_path).replace(" ", "_") if video_path else "unknown_video"
    filename = os.path.join(outfolder, f"{subtype_normal}_{safe_video}_{tid}_{safe_label}_{ts}.jpg")
    try:
        roi_full = frame[y1:y2, x1:x2]
        cv2.imwrite(filename, roi_full)
        print(f"[INFO] Saved {violation_type} snapshot: {filename}")

        reference = f"track_{tid}"
        speed_val = current_speed.get(tid) if tid in current_speed else None

        # violation detection
        try:
            payload_v, detect_v = emit_alert_json_single_detection_snake(
                frame=frame,
                bbox=[x1, y1, x2, y2],
                violation_subtype=subtype_normal,
                reference=reference,
                speed=speed_val,
                color=color_label,
                details=f"{violation_type} detected",
                frame_location=video_path,
                vehicle_type=vehicle_type,
                frame_id=frame_id,
                zone_id=ZONE_ID,
                location_id=LOCATION_ID,
                analysis_type=FRAME_PAYLOAD_TYPE
            )
            if payload_v is not None and detect_v is not None:
                try:
                    detect_v['imageLocation'] = filename
                    payload_v['detections'][0] = detect_v
                except Exception:
                    pass
                try:
                    write_frame_json(payload_v, detect_v)
                except Exception as e:
                    print("[WARN] write_frame_json failed in save_violation_snapshot (violation):", e)
        except Exception as e:
            print("[WARN] Failed to emit/merge violation detection:", e)

        # companion vehicle detection
        try:
            vlabel = vehicle_type if (vehicle_type is not None and str(vehicle_type).strip()) else "unknown"
            payload_c, detect_c = emit_alert_json_single_detection_snake(
                frame=frame,
                bbox=[x1, y1, x2, y2],
                violation_subtype=vlabel,
                reference=reference,
                speed=speed_val,
                color=color_label,
                details="vehicle_detected_as_part_of_violation",
                frame_location=video_path,
                vehicle_type=vlabel,
                frame_id=frame_id,
                zone_id=ZONE_ID,
                location_id=LOCATION_ID,
                analysis_type=FRAME_PAYLOAD_TYPE
            )
            if payload_c is not None and detect_c is not None:
                try:
                    write_frame_json(payload_c, detect_c)
                except Exception as e:
                    print("[WARN] write_frame_json failed in save_violation_snapshot (companion vehicle):", e)
        except Exception as e:
            print("[WARN] Failed to emit/merge companion vehicle detection:", e)

        return filename
    except Exception as e:
        print("[WARN] Failed to save snapshot:", e)
        return None

def save_density_snapshot_and_alert(frame, density, video_path=None, frame_id=None, total_count=None):
    outfolder = os.path.join("outputs", "occupancy")
    try:
        os.makedirs(outfolder, exist_ok=True)
    except:
        pass
    ts = time.strftime('%Y%m%d_%H%M%S')
    safe_video = os.path.basename(video_path).replace(" ", "_") if video_path else "unknown_video"
    filename = os.path.join(outfolder, f"occupancy_{safe_video}_{int(time.time())}_{ts}.jpg")
    try:
        # Draw total vehicle count on frame if provided
        try:
            if total_count is not None:
                cv2.putText(frame, f"Total vehicles (zone): {int(total_count)}",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        except Exception:
            pass

        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved occupancy snapshot: {filename}")

        h,w = frame.shape[:2]
        bbox = [0,0,w-1,h-1]
        reference = f"density_{int(time.time())}"

        # Use density as the numeric value in the detection payload.
        payload, detect_obj = emit_alert_json_single_detection_snake(
            frame=frame,
            bbox=bbox,
            violation_subtype="occupancy",
            reference=reference,
            speed=density,   # keep backward compatibility; this becomes the 'value'
            color=None,
            details=f"Density alert: {density:.2f}",
            frame_location=video_path,
            vehicle_type=None,
            frame_id=frame_id,
            zone_id=ZONE_ID,
            location_id=LOCATION_ID,
            analysis_type=FRAME_PAYLOAD_TYPE
        )
        if payload is None or detect_obj is None:
            # If emitter failed, still return filename
            return filename

        # Add explicit totalVehicles field so consumers can read exact count
        try:
            detect_obj['totalVehicles'] = int(total_count) if total_count is not None else None
            # ensure detect imageLocation is attached
            detect_obj['imageLocation'] = filename
            payload['detections'][0] = detect_obj
        except Exception:
            pass

        try:
            write_frame_json(payload, detect_obj)
        except Exception as e:
            print("[WARN] write_frame_json failed in save_density_snapshot_and_alert:", e)

        return filename
    except Exception as e:
        print("[WARN] save_density_snapshot failed:", e)
        return None


# --------------------------- Simple UI-less add_stream_url helper ---------------------------
def add_stream_url_headless(p):
    """
    Set the selected_videos to single entry for headless execution.
    """
    global selected_videos
    selected_videos = [{'path': p, 'loc': video_location_default}]
    video_locations.clear()
    video_locations[p] = video_location_default
    print(f"[DEBUG] Stream added in headless mode: {p}")

# The rest of the code (ROI binding, UI widgets) is intentionally left intact as functions,
# but they will not be used in headless mode.

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = max(0, (box1[2]-box1[0])) * max(0, (box1[3]-box1[1]))
    a2 = max(0, (box2[2]-box2[0])) * max(0, (box2[3]-box2[1]))
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / (union + 1e-9)

def check_wrong_way_and_density(tid, cy, lanes_count, expected_direction):
    y_history[tid].append(cy)
    if len(y_history[tid]) > 5:
        y_history[tid] = y_history[tid][-5:]
    moving_toward = moving_away = False
    if len(y_history[tid]) >= 3:
        if y_history[tid][-3] < y_history[tid][-2] < y_history[tid][-1]:
            moving_toward = True
        if y_history[tid][-3] > y_history[tid][-2] > y_history[tid][-1]:
            moving_away = True
    wrong = False
    if expected_direction == "in" and moving_away:
        wrong = True
    if expected_direction == "out" and moving_toward:
        wrong = True
    zone_flag = cy > 200
    return wrong, zone_flag

# --------------------------- MAIN MONITOR LOOP (unchanged) ---------------------------
def monitor_traffic(video_path, lanes_count, expected_direction, video_location=None, ui_key=None):
    global running, car_count, bus_count, truck_count, motorcycle_count

    if not video_path:
        if file_label:
            try:
                file_label.config(text="Please select a video file first!", fg="red")
            except:
                pass
        print("[ERROR] No video selected.")
        return

    model = get_shared_model()
    if model is None:
        print("[ERROR] Shared model not available.")
        if file_label:
            try:
                file_label.config(text="Model load error", fg="red")
            except:
                pass
        return

    try:
        classnames = model.names
    except Exception:
        classnames = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

    safe_path = str(video_path)
    print(f"[INFO] Opening video/stream: {safe_path}")
    try:
        cap = cv2.VideoCapture(safe_path)
    except:
        cap = cv2.VideoCapture(safe_path)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(safe_path, cv2.CAP_FFMPEG)
        except:
            pass
    if not cap.isOpened():
        print("[ERROR] Cannot open video/stream:", safe_path)
        if file_label:
            try:
                file_label.config(text="Failed to open video!", fg="red")
            except:
                pass
        return

    tracker = None
    if Sort is not None:
        try:
            tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
        except Exception as e:
            print("[WARN] Failed to create Sort tracker:", e)
            tracker = None
    else:
        print("[WARN] Sort not available — using per-frame pseudo-tracking fallback.")

    vehicle_count_set.clear()
    car_count = bus_count = truck_count = 0
    motorcycle_count = 0

    VIDEO_H = PANEL_H
    writer = None
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
    except:
        src_fps = 20.0
    out_filename = os.path.join(outputs_dir, f"output_{int(time.time())}.mp4")
    print(f"[INFO] Output will be saved to: {out_filename} (fps={src_fps})")

    frame_fail_count = 0
    last_cleanup_ts = time.time()
    frame_index = 0
    inference_count = 0

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                frame_fail_count += 1
                if frame_fail_count >= FRAME_FAILURE_RETRY_LIMIT:
                    print("[WARN] Reached frame failure limit - ending loop")
                    break
                time.sleep(0.05)
                continue
            frame_fail_count = 0

            frame_index += 1

            # Build deterministic frame_id for this frame (use ms timestamp + frame index)
            tsms = int(time.time()*1000)
            base = os.path.splitext(os.path.basename(video_path))[0]
            frame_id = f"{base}_frame_{frame_index}_{tsms}"

            # --- SKIP frames to reduce load (process every 4th frame) ---
            if frame_index % 5 != 0:
                # push a resized preview into _frame_buf for imshow
                try:
                    video_w = 800
                    frame_small = cv2.resize(frame, (video_w, VIDEO_H))
                except:
                    frame_small = frame.copy()
                try:
                    with _frame_lock:
                        _frame_buf.append(frame_small)
                except:
                    pass
                continue

            inference_count += 1
            try:
                results = list(model(frame, stream=False, device=DEVICE))
            except TypeError:
                results = list(model(frame, stream=False))
            except Exception:
                traceback.print_exc()
                continue

            dets = np.empty((0,5), dtype=np.float32)
            detected_vehicles = {}
            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                try:
                    xyxys = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
                    clss  = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
                    for xyxy, conf, cls_idx in zip(xyxys, confs, clss):
                        x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
                        conf = float(conf)
                        cls_idx = int(cls_idx)
                        label = classnames.get(cls_idx, str(cls_idx)) if isinstance(classnames, dict) else (classnames[cls_idx] if cls_idx < len(classnames) else str(cls_idx))
                        label = label.lower() if isinstance(label, str) else str(label)
                        if label not in ["car","bus","truck","motorbike","motorcycle"]:
                            continue
                        if conf < DET_CONF or (x2-x1)*(y2-y1) < 400:
                            continue
                        if "motor" in label:
                            label = "motorcycle"

                        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                        poly = roi_polygons.get(video_path)
                        if poly:
                            if not point_in_polygon(cx, cy, poly):
                                continue
                        else:
                            roi = roi_boxes.get(video_path)
                            if roi is not None:
                                rx1, ry1, rx2, ry2 = roi
                                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                                    continue

                        det = np.array([x1,y1,x2,y2,conf], dtype=np.float32)
                        dets = np.vstack((dets, det))
                        detected_vehicles[(x1,y1,x2,y2)] = label
                except Exception:
                    for box in boxes:
                        try:
                            conf = float(getattr(box, "conf", box[4]))
                            cls_idx = int(getattr(box, "cls", box[5]))
                        except Exception:
                            continue
                        label = classnames.get(cls_idx, str(cls_idx)) if isinstance(classnames, dict) else (classnames[cls_idx] if cls_idx < len(classnames) else str(cls_idx))
                        label = label.lower() if isinstance(label, str) else str(label)
                        try:
                            coords = box.xyxy[0].tolist()
                        except:
                            try:
                                coords = list(box.xyxy)
                            except:
                                continue
                        try:
                            x1,y1,x2,y2 = map(int, coords[:4])
                        except:
                            continue
                        if "motor" in label:
                            label = "motorcycle"
                        if label in ['car','truck','bus','motorcycle'] and conf >= DET_CONF:
                            roi = roi_boxes.get(video_path)
                            if roi is not None:
                                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                                rx1, ry1, rx2, ry2 = roi
                                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                                    continue
                            det = np.array([x1,y1,x2,y2,conf], dtype=np.float32)
                            dets = np.vstack((dets, det))
                            detected_vehicles[(x1,y1,x2,y2)] = label

            tracks = []
            if tracker is not None:
                try:
                    tracks = tracker.update(dets)
                except Exception:
                    traceback.print_exc()
                    tracks = []
            else:
                for idx, d in enumerate(dets):
                    x1,y1,x2,y2,conf = [int(float(v)) for v in d.tolist()]
                    tracks.append([x1,y1,x2,y2, idx+1])

            try:
                tracks = np.array(tracks, dtype=np.int32)
            except:
                pass

            current_tracks_boxes = {}
            current_tracks_labels = {}
            zone_count = 0
            now_ts = time.time()
            h,w = frame.shape[:2]

            # First pass
            for tr in tracks:
                try:
                    x1,y1,x2,y2,tid = map(int, tr)
                except:
                    vals = [int(float(v)) for v in tr]
                    x1,y1,x2,y2,tid = vals

                display_id = get_display_id(video_path, tid)
                cx = (x1+x2)//2; cy = (y1+y2)//2
                now = now_ts
                best_i = 0.0; best_label = None
                tbox = [x1,y1,x2,y2]
                for (dx1,dy1,dx2,dy2), lab in detected_vehicles.items():
                    iv = iou([dx1,dy1,dx2,dy2], tbox)
                    if iv > best_i:
                        best_i = iv; best_label = lab
                vehicle_type_for_json = best_label
                vehicle_type_ui = best_label if best_i >= LABEL_IOU_THRESH else None
                current_tracks_labels[tid] = vehicle_type_ui
                seen_frames[tid] = seen_frames.get(tid,0) + 1
                last_seen_ts[tid] = now

                if vehicle_type_ui:
                    if seen_frames[tid] >= SEEN_FRAMES_MIN and tid not in counted_ids:
                        counted_ids.add(tid)
                        vehicle_count_set.add(tid)
                        try:
                            src = os.path.basename(video_path)
                            # in UI mode log_vehicle would add to tree; headless we just print
                            print(f"[LOG] Vehicle ID:{display_id} Type:{vehicle_type_ui} Source:{src}")
                        except:
                            pass
                        if vehicle_type_ui == 'car': car_count += 1
                        elif vehicle_type_ui == 'bus': bus_count += 1
                        elif vehicle_type_ui == 'truck': truck_count += 1
                        elif vehicle_type_ui == 'motorcycle': motorcycle_count += 1

                   # ------------- REPLACEMENT: robust speed calculation WITHOUT freeze-row -------------
                    # configuration (tweak if needed)
                    MIN_MOVEMENT_PIXELS = 3.0        # ignore tiny jitter
                    MAX_JUMP_PIXELS = 200.0         # ignore unrealistic single-frame teleports
                    MIN_DT_SECONDS = 1e-3           # clamp tiny dt to avoid huge speeds
                    MAX_POSSIBLE_SPEED_KMH = 130.0  # sanity clamp for vehicles
                    SPEED_SMOOTHING_ALPHA = 0.4     # EMA alpha: 0 = no smoothing, 1 = instant value

                    if tid in prev_positions:
                        # prev_positions may be (px,py,pt) or (px,py,pt,prev_speed)
                        prev_entry = prev_positions[tid]
                        try:
                            if len(prev_entry) >= 4:
                                px, py, pt, prev_speed = prev_entry
                            else:
                                px, py, pt = prev_entry
                                prev_speed = None
                        except Exception:
                            px, py, pt = prev_entry if isinstance(prev_entry, (list,tuple)) and len(prev_entry)>=3 else (cx, cy, now)
                            prev_speed = None

                        dist_px = math.hypot(cx - px, cy - py)

                        # compute dt in seconds — keep simple (we assume pt is seconds)
                        dt = now - float(pt) if isinstance(pt, (int,float)) else (1.0 / (src_fps or 20.0))
                        if dt <= MIN_DT_SECONDS:
                            dt = MIN_DT_SECONDS

                        # ignore tiny jitter
                        if dist_px < MIN_MOVEMENT_PIXELS:
                            speed_kmh_inst = prev_speed if prev_speed is not None else 0.0
                        else:
                            # ignore impossible teleports
                            if dist_px > MAX_JUMP_PIXELS:
                                speed_kmh_inst = prev_speed if prev_speed is not None else 0.0
                            else:
                                try:
                                    speed_m_s = (dist_px * PIXEL_TO_METER) / dt
                                    speed_kmh_inst = speed_m_s * 3.6 * CALIBRATION_FACTOR
                                except Exception:
                                    speed_kmh_inst = prev_speed if prev_speed is not None else 0.0

                                # sanity clamp
                                if speed_kmh_inst < 0:
                                    speed_kmh_inst = 0.0
                                elif speed_kmh_inst > MAX_POSSIBLE_SPEED_KMH:
                                    speed_kmh_inst = MAX_POSSIBLE_SPEED_KMH

                        # exponential smoothing
                        if prev_speed is None:
                            smoothed_speed = speed_kmh_inst
                        else:
                            smoothed_speed = (SPEED_SMOOTHING_ALPHA * speed_kmh_inst +
                                            (1.0 - SPEED_SMOOTHING_ALPHA) * prev_speed)

                        # ALWAYS use the smoothed current speed (no freeze-row)
                        current_speed[tid] = smoothed_speed

                        speed_kmh = current_speed[tid]
                        try:
                            speed_limit = int(speed_limit_var.get())
                        except:
                            speed_limit = SPEED_LIMIT

                        if speed_kmh > speed_limit:
                            try:
                                cx1,cy1,cx2,cy2 = crop_inner_box(x1,y1,x2,y2)
                                roi = frame[max(0,cy1):max(0,cy2), max(0,cx1):max(0,cx2)]
                                color_label = detect_roi_color_label(roi)
                            except:
                                color_label = "unknown"
                            try:
                                src = os.path.basename(video_path)
                                print(f"[VIOLATION] Speed ID:{display_id} {speed_kmh:.1f} Source:{src} Color:{color_label}")
                            except:
                                pass
                            if tid not in violation_timers:
                                violation_timers[tid] = time.time()
                                save_violation_snapshot(frame, [x1,y1,x2,y2], tid, "speed", color_label, video_path, vehicle_type_for_json, frame_id=frame_id)

                        # round for display
                        speed_kmh = round(speed_kmh)
                        if speed_kmh <= STOP_SPEED_THRESHOLD:
                            box_color = (0,0,0)
                        elif speed_kmh > speed_limit:
                            box_color = (0,0,255)
                        else:
                            box_color = (0,255,0)

                        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
                        cv2.putText(frame, f"{speed_kmh:.1f} km/h", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    else:
                        direction_history.setdefault(tid, deque(maxlen=10))
                        
                    current_tracks_boxes[tid] = [x1, y1, x2, y2]
                    # store updated prev_positions as (cx,cy,pt,prev_speed)
                    prev_positions[tid] = (cx, cy, now, float(current_speed.get(tid, 0.0)))
                    # ------------- END REPLACEMENT --------------------------------------------



            # Add per-frame vehicle detections so frame_<frame_id>.json contains every vehicle
            try:
                for tid, box in current_tracks_boxes.items():
                    try:
                        x1, y1, x2, y2 = box
                        vehicle_subtype = current_tracks_labels.get(tid) or "unknown"
                        reference = f"track_{tid}"
                        speed_val = current_speed.get(tid)

                        payload, detect_obj = emit_alert_json_single_detection_snake(
                            frame=frame,
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            violation_subtype="vehicle_type",
                            reference=reference,
                            speed=speed_val,
                            color=None,
                            details="vehicle_detected",
                            frame_location=video_path,
                            vehicle_type=vehicle_subtype,
                            frame_id=frame_id,
                            zone_id=ZONE_ID,
                            location_id=LOCATION_ID,
                            analysis_type=FRAME_PAYLOAD_TYPE
                        )
                        if payload is not None and detect_obj is not None:
                            try:
                                write_frame_json(payload, detect_obj)
                            except Exception as e:
                                print("[WARN] write_frame_json failed for per-track detection:", e)
                    except Exception:
                        pass
            except Exception as e:
                print("[WARN] Per-frame detection emission failed:", e)

            # Second pass (wrong-way detection, drawing, etc.)
            for tr in tracks:
                try:
                    x1,y1,x2,y2,tid = map(int, tr)
                except:
                    vals = [int(float(v)) for v in tr]
                    x1,y1,x2,y2,tid = vals

                display_id = get_display_id(video_path, tid)
                cx = (x1+x2)//2; cy = (y1+y2)//2
                now = now_ts
                speed_kmh = current_speed.get(tid, 0.0)
                box_diag = math.hypot(x2-x1, y2-y1)
                if box_diag <= 0:
                    box_diag = float(200)
                dynamic_radius = int(max(200*0.5, min(200*2.0, box_diag*1.2)))
                neighbors = 0
                for oid, obox in current_tracks_boxes.items():
                    if oid == tid: continue
                    ox1,oy1,ox2,oy2 = obox
                    ocx = (ox1+ox2)//2; ocy = (oy1+oy2)//2
                    if math.hypot(cx-ocx, cy-ocy) < dynamic_radius:
                        neighbors += 1

                wrong, in_zone = check_wrong_way_and_density(tid, cy, lanes_count, expected_direction)
                if wrong:
                    try:
                        cx1,cy1,cx2,cy2 = crop_inner_box(x1,y1,x2,y2)
                        roi = frame[max(0,cy1):max(0,cy2), max(0,cx1):max(0,cx2)]
                        color_label = detect_roi_color_label(roi)
                    except:
                        color_label = "unknown"
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(frame, f"WRONG WAY ID:{display_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    try:
                        src = os.path.basename(video_path)
                        print(f"[VIOLATION] Wrong Way ID:{display_id} Color:{color_label} Source:{src}")
                    except:
                        pass
                    known_vehicle_type = current_tracks_labels.get(tid) or None
                    if known_vehicle_type is None:
                        best_i = 0.0; best_label = None
                        tbox = [x1,y1,x2,y2]
                        for (dx1,dy1,dx2,dy2), lab in detected_vehicles.items():
                            iv = iou([dx1,dy1,dx2,dy2], tbox)
                            if iv > best_i:
                                best_i = iv; best_label = lab
                        known_vehicle_type = best_label
                    save_violation_snapshot(frame, [x1,y1,x2,y2], tid, "wrongway", color_label, video_path, known_vehicle_type, frame_id=frame_id)
                else:
                    box_color = (255,255,0)
                    best_i = 0.0; best_label = None
                    tbox = [x1,y1,x2,y2]
                    for (dx1,dy1,dx2,dy2), lab in detected_vehicles.items():
                        iv = iou([dx1,dy1,dx2,dy2], tbox)
                        if iv > best_i:
                            best_i = iv; best_label = lab
                    vtype = best_label if best_i >= LABEL_IOU_THRESH else None
                    if vtype == "motorcycle": box_color = (0,165,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
                    cv2.putText(frame, f"ID:{display_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                if in_zone:
                    zone_count += 1

            density = (zone_count / 1) 
            cv2.putText(frame, f"Vehicles occupancy:{zone_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            if density >= 10.0:
                cv2.putText(frame, "\u26A0 TRAFFIC WARNING", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
                # use zone_count (vehicles in the zone) as the total vehicle count for occupancy alerts
                snapshot_file = save_density_snapshot_and_alert(frame, density, video_path, frame_id=frame_id, total_count=zone_count)

                try:
                    loc = video_locations.get(video_path, video_location_default)
                except:
                    loc = video_location_default
                plot_event_on_map(loc, "Traffic Warning", f"density={density:.2f}", snapshot_file)

            try:
                vehicle_count_var.set(len(vehicle_count_set))
                car_count_var.set(car_count)
                bus_count_var.set(bus_count)
                truck_count_var.set(truck_count)
                motorcycle_count_var.set(motorcycle_count)
            except:
                pass

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_size = (frame.shape[1], frame.shape[0])
                try:
                    writer = cv2.VideoWriter(out_filename, fourcc, src_fps, out_size)
                    if not writer.isOpened():
                        print("[WARN] VideoWriter failed to open — check codecs. Output disabled.")
                        writer = None
                except Exception as e:
                    print("[WARN] Failed to create VideoWriter:", e)
                    writer = None

            if writer is not None:
                try:
                    writer.write(frame)
                except Exception as e:
                    print("[WARN] Failed to write frame to output:", e)

            # push frame to preview buffer (_frame_buf) for cv.imshow
            try:
                frame_for_ui = cv2.resize(frame, (max(320, int(frame.shape[1] * 0.5)), VIDEO_H))
                try:
                    with _frame_lock:
                        _frame_buf.append(frame_for_ui)
                except:
                    pass
            except:
                pass

            # periodic cleanup
            if time.time() - last_cleanup_ts > 2.0:
                stale = []
                nowc = time.time()
                for tid, ts in list(last_seen_ts.items()):
                    if nowc - ts > COUNT_COOLDOWN_SEC:
                        stale.append(tid)
                for tid in stale:
                    seen_frames.pop(tid, None)
                    last_seen_ts.pop(tid, None)
                    prev_positions.pop(tid, None)
                    current_speed.pop(tid, None)
                    frozen_speed.pop(tid, None)
                    direction_history.pop(tid, None)
                    vehicle_count_set.discard(tid)
                    counted_ids.discard(tid)
                    y_history.pop(tid, None)
                last_cleanup_ts = time.time()

    finally:
        try: cap.release()
        except: pass
        try:
            if writer is not None:
                writer.release()
                print(f"[INFO] Saved output video to: {out_filename}")
        except:
            pass
        print("[INFO] Monitor loop ended.")
        try: cap.release()
        except: pass
        print(f"[INFO] Total frames read: {frame_index}")
        print(f"[INFO] Frames with inference: {inference_count}")

# --------------------------- CONTROL HANDLERS (kept, adapted) ---------------------------
def start_monitoring():
    global running, lanes_user_count, expected_direction_global, ZONE_ID, LOCATION_ID, FRAME_PAYLOAD_TYPE
    if not selected_videos:
        print("[ERROR] No video/stream selected. Use --url or --video.")
        return

    try:
        lanes_user_count = int(lanes_entry.get())
        if lanes_user_count <= 0:
            lanes_user_count = 3
    except Exception:
        lanes_user_count = 3

    expected_direction_global = direction_var.get().strip().lower()
    if expected_direction_global not in ("in", "out"):
        expected_direction_global = "in"

    try:
        zid_val = zone_id_var.get().strip()
        if zid_val == "":
            ZONE_ID = None
        else:
            try:
                ZONE_ID = int(zid_val)
            except:
                ZONE_ID = zid_val
    except Exception:
        ZONE_ID = None

    try:
        lid_val = location_id_var.get().strip()
        if lid_val == "":
            LOCATION_ID = None
        else:
            try:
                LOCATION_ID = int(lid_val)
            except:
                LOCATION_ID = lid_val
    except Exception:
        LOCATION_ID = None

    print(f"[INFO] Starting monitoring. lanes={lanes_user_count} direction={expected_direction_global} ZONE_ID={ZONE_ID} LOCATION_ID={LOCATION_ID}")

    running = True

    v = selected_videos[0]
    path = v.get('path'); loc = v.get('loc')
    video_locations[path] = loc

    threading.Thread(target=monitor_traffic, args=(path, lanes_user_count, expected_direction_global, loc, path), daemon=True).start()

def stop_monitoring():
    global running
    running = False

# --------------------------- Headless preview loop (cv.imshow) ---------------------------
_frame_buf = deque(maxlen=4)
_frame_lock = threading.Lock()

def headless_preview_loop(window_name="SmartTrafficPreview"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    try:
        while running:
            frame = None
            try:
                with _frame_lock:
                    if len(_frame_buf) > 0:
                        frame = _frame_buf.pop(); _frame_buf.clear()
            except:
                frame = None
            if frame is not None:
                try:
                    cv2.imshow(window_name, frame)
                except Exception as e:
                    print("[WARN] imshow error:", e)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Exit key pressed.")
                stop_monitoring()
                break
            time.sleep(0.01)
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass

# --------------------------- CLI entrypoint ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Behavior Monitoring (no UI)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Stream URL (RTSP/RTMP/HTTP) or camera index (e.g. 0)")
    group.add_argument("--video", help="Path to video file")
    parser.add_argument("--lanes", type=int, default=3, help="Number of lanes for density calculation")
    parser.add_argument("--direction", choices=["in","out"], default="in", help="Expected direction")
    parser.add_argument("--zone", help="Optional Zone ID")
    parser.add_argument("--location", help="Optional Location ID")
    # Don't reference MODEL_WEIGHTS here - use None default
    parser.add_argument("--model", help="YOLO model weights (defaults to yolo12x.pt)", default=None)
    args = parser.parse_args()

    # now safely set the global (if provided)
    global MODEL_WEIGHTS
    if args.model:
        MODEL_WEIGHTS = args.model

    source = args.url if args.url else args.video
    # if numeric camera index passed as string, keep as is for OpenCV (e.g. "0")
    add_stream_url_headless(source)

    lanes_entry.set(str(args.lanes))
    direction_var.set(args.direction)
    zone_id_var.set("" if args.zone is None else str(args.zone))
    location_id_var.set("" if args.location is None else str(args.location))

    start_monitoring()
    # run preview loop (blocks until stop)
    headless_preview_loop()

    # wait a bit for monitor threads to finish
    time.sleep(0.5)
    print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
