# server.py - FINAL FIXED & OPTIMIZED VERSION
import asyncio
import time
import json
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import pathlib

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from config import DEVICE
from frameProcessor import FrameProcessor
from loadMidas import loadMidas
from loadModel import loadModel

# ==================== LOAD MODELS ====================
midas, midas_transform = loadMidas()
main_model, focus_model = loadModel()

# ==================== SERVER ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame_processor")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    print(f"Client connected: {client_info}")

    processor = FrameProcessor(midas,midas_transform,main_model,focus_model)
    last_frame_time = time.time()

    try:
        await websocket.send_json({"type": "status", "msg": "Server ready! Processing frames..."})
    except Exception as e:
        print(f"Initial send error: {e}")
        return

    async def keepalive():
        try:
            while True:
                await asyncio.sleep(15.0)
                try:
                    await websocket.send_json({"type": "ping", "ts": time.time()})
                except:
                    break
        except:
            return

    ka_task = asyncio.create_task(keepalive())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "status", "msg": "Connection active"})
                    continue
                except:
                    break

            if "bytes" in msg:
                current_time = time.time()
                if current_time - last_frame_time < 0.1:
                    continue
                last_frame_time = current_time
                frame_bytes = msg["bytes"]

                if len(frame_bytes) < 1000:
                    continue

                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, processor.process, frame),
                    timeout=2.0
                )
                await websocket.send_json(result)

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("cmd") == "ping":
                        await websocket.send_json({"type": "pong"})
                except:
                    pass

    except WebSocketDisconnect:
        print(f"Client disconnected: {client_info}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        ka_task.cancel()
        processor.cleanup()
        try:
            await websocket.close(code=1000)
        except:
            pass
        print(f"Connection closed: {client_info}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": main_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ws_max_size=5 * 1024 * 1024
    )