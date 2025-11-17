import time
import gc
from typing import List, Dict, Any
import numpy as np
import cv2
import torch

from config import INFER_H,INFER_W,BUFFER_SIZE,MIN_SUGGESTION_INTERVAL,MIN_CONSECUTIVE_FRAMES, DOMINANT_THRESHOLD,STOP_OBJECTS,OBJECT_DANGER_THRESHOLD,CENTRAL_ZONE_END,CENTRAL_ZONE_START,DISPLAY_H,DISPLAY_W,DEVICE,DEPTH_EVERY,SMOOTHING_ALPHA,DETECT_EVERY,GENERAL_THRESHOLD,MIN_FREE_HEIGHT,MIN_FREE_WIDTH_PCT,FOCUS_CLASSES,FOCUS_THRESHOLD,MIN_SAFE_THRESHOLD,SAFE_HEAT_MAX,MAX_DETECTIONS

from helper import normalize_map,combine_maps,compute_optical_flow,box_mean_heat

class FrameProcessor:
    def __init__(self,midas,midas_transform,main_model,focus_model):
        self.prev_gray_small = None
        self.smoothed_heat_small = np.zeros((INFER_H, INFER_W), dtype=np.float32)
        self.frame_idx = 0
        self.last_suggestion = "Starting up..."
        self.last_zones = {"LEFT": 0.5, "FORWARD": 0.5, "RIGHT": 0.5}

        self.last_suggestion_time = time.time()
        self.suggestion_buffer = []
        self.buffer_size = BUFFER_SIZE
        self.current_stable_suggestion = "Starting up..."
        self.min_suggestion_interval = MIN_SUGGESTION_INTERVAL
        self.consecutive_count = 0
        self.min_consecutive_frames = MIN_CONSECUTIVE_FRAMES
        
        self.object_memory = []
        self.last_objects = []
        self.midas=midas
        self.midas_transform=midas_transform
        self.main_model=main_model
        self.focus_model=focus_model

    def get_stable_suggestion(self, new_suggestion: str, zones: Dict[str, float]) -> str:
        current_time = time.time()
        
        self.suggestion_buffer.append(new_suggestion)
        if len(self.suggestion_buffer) > self.buffer_size:
            self.suggestion_buffer.pop(0)
        
        suggestion_counts = {}
        for sug in self.suggestion_buffer:
            suggestion_counts[sug] = suggestion_counts.get(sug, 0) + 1
        
        most_frequent = max(suggestion_counts.items(), key=lambda x: x[1])
        dominant_suggestion, dominant_count = most_frequent
        
        time_since_last_change = current_time - self.last_suggestion_time
        
        if new_suggestion == self.current_stable_suggestion:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        should_change = (
            time_since_last_change >= self.min_suggestion_interval and
            (
                dominant_count >= len(self.suggestion_buffer) * DOMINANT_THRESHOLD or
                (self.consecutive_count >= self.min_consecutive_frames and 
                 zones.get(dominant_suggestion.split()[1] if "MOVE" in dominant_suggestion else "FORWARD", 1.0) < 0.7)
            )
        )
        
        if should_change:
            self.current_stable_suggestion = dominant_suggestion
            self.last_suggestion_time = current_time
            self.consecutive_count = 0
        
        return self.current_stable_suggestion

    def analyze_objects_for_suggestions(self, detections: List, zones: Dict, safest_zone: str) -> Dict:
        analysis = {
            "primary_suggestion": "",
            "object_warnings": [],
            "immediate_stop": False,
            "blocking_objects": []
        }
        
        dangerous_objects = []
        for det in detections:
            name = det["name"]
            conf = det["conf"]
            heat = det["heat"]
            box = det["box"]
            
            if name in STOP_OBJECTS and conf > 0.5:
                x_center = (box[0] + box[2]) / 2 / INFER_W  # Normalized to 0-1
                
                danger_level = "HIGH" if heat > OBJECT_DANGER_THRESHOLD else "MEDIUM"
                
                obj_info = {
                    "name": name,
                    "confidence": conf,
                    "danger_level": danger_level,
                    "heat_value": heat,
                    "position": "CENTER" if CENTRAL_ZONE_START <= x_center <= CENTRAL_ZONE_END else "SIDE",
                    "x_center": x_center
                }
                dangerous_objects.append(obj_info)
                
                if danger_level == "HIGH" and obj_info["position"] == "CENTER":
                    analysis["immediate_stop"] = True
                    analysis["blocking_objects"].append(obj_info)
        
        dangerous_objects.sort(key=lambda x: x["heat_value"], reverse=True)
        
        if analysis["immediate_stop"]:
            main_obj = analysis["blocking_objects"][0]
            analysis["primary_suggestion"] = f"STOP! {main_obj['name'].upper()} BLOCKING PATH"
            analysis["object_warnings"] = [f"Blocked by {main_obj['name']} (danger: {main_obj['danger_level']})"]
        
        elif dangerous_objects:
            main_obj = dangerous_objects[0]
            
            if main_obj["position"] == "SIDE":
                if main_obj["x_center"] < 0.5:  # Object on left
                    if safest_zone != "LEFT" or zones["RIGHT"] < zones["LEFT"] + 0.1:
                        analysis["primary_suggestion"] = f"MOVE RIGHT - {main_obj['name'].upper()} ON LEFT"
                    else:
                        analysis["primary_suggestion"] = f"CAUTION - {main_obj['name'].upper()} ON LEFT"
                else:  # Object on right
                    if safest_zone != "RIGHT" or zones["LEFT"] < zones["RIGHT"] + 0.1:
                        analysis["primary_suggestion"] = f"MOVE LEFT - {main_obj['name'].upper()} ON RIGHT"
                    else:
                        analysis["primary_suggestion"] = f"CAUTION - {main_obj['name'].upper()} ON RIGHT"
            else:
                analysis["primary_suggestion"] = f"CAUTION - {main_obj['name'].upper()} AHEAD"
            
            for obj in dangerous_objects[:3]:
                analysis["object_warnings"].append(
                    f"{obj['name']} ({obj['position'].lower()}, danger: {obj['danger_level']})"
                )
        
        else:
            analysis["primary_suggestion"] = "" 
        
        return analysis

    def get_enhanced_suggestion(self, raw_suggestion: str, zones: Dict, safest_zone: str, detections: List) -> str:
        """Combine path analysis with object awareness"""
        object_analysis = self.analyze_objects_for_suggestions(detections, zones, safest_zone)
        
        if object_analysis["immediate_stop"]:
            return object_analysis["primary_suggestion"]
        
        if object_analysis["primary_suggestion"] and "CAUTION" in object_analysis["primary_suggestion"]:
            return object_analysis["primary_suggestion"]
        
        if object_analysis["primary_suggestion"] and "MOVE" in object_analysis["primary_suggestion"]:
            return object_analysis["primary_suggestion"]
        
        # Fall back to path-based suggestion
        return raw_suggestion

    def cleanup(self):
        self.prev_gray_small = None
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    def process(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        try:
            start_time = time.time()

            # Resize
            frame_disp = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))
            frame_small = cv2.resize(frame_disp, (INFER_W, INFER_H))
            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            # === DEPTH === 
            depth_norm = np.zeros((INFER_H, INFER_W), dtype=np.float32)
            if self.frame_idx % DEPTH_EVERY == 0:
                try:
                    input_batch = self.midas_transform(frame_small).to(DEVICE)
                    with torch.no_grad():
                        pred = self.midas(input_batch)
                        pred = torch.nn.functional.interpolate(
                            pred.unsqueeze(1),
                            size=(INFER_H, INFER_W),
                            mode="bicubic",
                            align_corners=False
                        ).squeeze().cpu().numpy()
                    depth_norm = normalize_map(pred)
                except Exception as e:
                    print(f"Depth error: {e}")

            # === OPTICAL FLOW ===
            flow_norm = np.zeros_like(gray_small, dtype=np.float32)
            if self.prev_gray_small is not None:
                try:
                    flow_norm = compute_optical_flow(self.prev_gray_small, gray_small)
                except Exception as e:
                    print(f"Flow error: {e}")
            self.prev_gray_small = gray_small.copy()

            # === COMBINE & SMOOTH ===
            combined = combine_maps(depth_norm, flow_norm)
            self.smoothed_heat_small = (
                SMOOTHING_ALPHA * self.smoothed_heat_small +
                (1.0 - SMOOTHING_ALPHA) * combined
            )

            # === YOLO DETECTIONS ===
            detections_small = []
            if self.frame_idx % DETECT_EVERY == 0:
                try:
                    results = [
                        self.main_model.predict(frame_small, imgsz=max(INFER_W, INFER_H), device=DEVICE,
                                        verbose=False, conf=GENERAL_THRESHOLD),
                        self.focus_model.predict(frame_small, imgsz=max(INFER_W, INFER_H), device=DEVICE,
                                        verbose=False, conf=FOCUS_THRESHOLD)
                    ]

                    temp = []
                    for i, res in enumerate(results):
                        focused = (i == 1)
                        if res[0].boxes is not None:
                            for b in res[0].boxes:
                                xyxy = b.xyxy[0].cpu().numpy()
                                conf = float(b.conf[0])
                                cls_id = int(b.cls[0])
                                name = res[0].names[cls_id]
                                if not focused or name in FOCUS_CLASSES:
                                    temp.append((xyxy, conf, name, focused))

                    # Deduplication
                    final = []
                    for det in temp:
                        keep = True
                        x1, y1, x2, y2 = det[0]
                        for prev in final[:]:
                            px1, py1, px2, py2 = prev[0]
                            if det[2] == prev[2]:  # same class
                                iou = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
                                area1 = (x2-x1)*(y2-y1)
                                area2 = (px2-px1)*(py2-py1)
                                if iou > 0.7 * min(area1, area2):
                                    if det[1] > prev[1]:
                                        final.remove(prev)
                                    else:
                                        keep = False
                        if keep:
                            final.append(det)

                    detections_small = [(x1, y1, x2, y2, c, n, f)
                                      for (xyxy, c, n, f) in final
                                      for x1, y1, x2, y2 in [xyxy]]

                except Exception as e:
                    print(f"Detection error: {e}")

            # === ZONE ANALYSIS ===
            h, w = self.smoothed_heat_small.shape
            zone_w = w // 3

            left_avg = float(self.smoothed_heat_small[:, :zone_w].mean())
            center_avg = float(self.smoothed_heat_small[:, zone_w:2*zone_w].mean())
            right_avg = float(self.smoothed_heat_small[:, 2*zone_w:].mean())
            
            zones = {"LEFT": left_avg, "FORWARD": center_avg, "RIGHT": right_avg}
            safest_zone = min(zones, key=zones.get)

            # Fallback: if all zones too hot
            if min(zones.values()) > MIN_SAFE_THRESHOLD:
                safest_zone = "FORWARD"

            self.last_zones = zones

            # === SAFE PATH ANALYSIS ===
            zone_slice = (
                slice(0, zone_w) if safest_zone == "LEFT" else
                slice(zone_w, 2*zone_w) if safest_zone == "FORWARD" else
                slice(2*zone_w, w)
            )

            zone_heat = self.smoothed_heat_small[:, zone_slice]
            
            # Safe path calculation
            free_height = np.zeros(zone_w, dtype=int)
            for col in range(zone_w):
                height = 0
                for row in range(h-1, -1, -1):
                    if zone_heat[row, col] < SAFE_HEAT_MAX:
                        height += 1
                    else:
                        break
                free_height[col] = height

            # Find widest continuous safe path
            max_w = 0
            best_start = 0
            curr_w = 0
            for col in range(zone_w):
                if free_height[col] >= MIN_FREE_HEIGHT:
                    curr_w += 1
                    if curr_w > max_w:
                        max_w, best_start = curr_w, col - curr_w + 1
                else:
                    curr_w = 0

            required_w = int(zone_w * MIN_FREE_WIDTH_PCT)
            has_safe_path = max_w >= required_w and zones[safest_zone] < MIN_SAFE_THRESHOLD

            # === DETECTION INFO ===
            detection_info = []
            for (x1, y1, x2, y2, conf, name, focused) in detections_small[:MAX_DETECTIONS]:
                heat = box_mean_heat(self.smoothed_heat_small, (x1, y1, x2, y2))
                detection_info.append({
                    "name": name,
                    "conf": float(conf),
                    "heat": float(heat),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "focused": bool(focused)
                })

            # === GENERATE ENHANCED SUGGESTION ===
            raw_suggestion = f"MOVE {safest_zone} - SAFE PATH!" if has_safe_path else "STOP - NO SAFE PATH!"            
            enhanced_suggestion = self.get_enhanced_suggestion(raw_suggestion, zones, safest_zone, detection_info)
            stable_suggestion = self.get_stable_suggestion(enhanced_suggestion, zones)
            self.last_objects = [det["name"] for det in detection_info[:3]]

            # === DEBUG LOG ===
            if self.frame_idx % 20 == 0:
                object_str = ", ".join(self.last_objects) if self.last_objects else "none"
                print(f"SMART | Final: {stable_suggestion} | "
                      f"Objects: {object_str} | "
                      f"Heat: L{left_avg:.2f} C{center_avg:.2f} R{right_avg:.2f}")

            processing_time = time.time() - start_time
            fps = 1.0 / (processing_time + 1e-8)

            self.frame_idx += 1
            self.last_suggestion = stable_suggestion

            if self.frame_idx % 100 == 0:
                self.cleanup()

            return {
                "suggestion": stable_suggestion,
                "has_safe_path": has_safe_path,
                "safest_zone": safest_zone,
                "zones": zones,
                "detections": detection_info,
                "fps": round(fps, 1),
                "error": False,
                "object_context": {
                    "detected_objects": self.last_objects,
                    "total_objects": len(detection_info),
                    "raw_suggestion": raw_suggestion
                }
            }

        except Exception as e:
            print(f"Processor error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "suggestion": self.last_suggestion,
                "has_safe_path": False,
                "safest_zone": "FORWARD",
                "zones": self.last_zones,
                "detections": [],
                "fps": 0,
                "error": True
            }
