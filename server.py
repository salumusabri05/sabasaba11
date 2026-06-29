import asyncio
import json
import base64
import os
import time
from typing import List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import cv2

from pipeline import SignPredictor, TextBuilder, CameraMetadata

app = FastAPI(title="Bridging Silence - TSL Word Recognition API")

origins = [
    "http://localhost:3000",
    "https://www.bridgingsilence.org",
    "https://bridgingsilence.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production, allow all for compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model initialization ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "tsl_word_model_11.h5")
print(f"Loading word recognition model from {MODEL_PATH}...")
predictor = SignPredictor(MODEL_PATH)
print("Model loaded successfully!")

# --- Azure Speech Configuration ---
try:
    import azure.cognitiveservices.speech as speechsdk
    speech_key = os.environ.get("AZURE_SPEECH_KEY")
    service_region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
    if not speech_key:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "sw-KE-ZuriNeural"
    # Set audio_config to None to synthesize to memory (audio_data bytes)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    print("Azure Speech synthesizer initialized.")
except (ImportError, Exception) as e:
    print(f"Failed to initialize Azure Speech synthesizer: {e}")
    synthesizer = None
    speechsdk = None

# --- Static files ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Pydantic Schemas ---
class Landmark(BaseModel):
    x: float
    y: float
    z: float

class FrameLandmarks(BaseModel):
    pose: Optional[List[Landmark]] = Field(default=None, description="List of 5 pose landmarks")
    left_hand: Optional[List[Landmark]] = Field(default=None, description="List of 21 left hand landmarks")
    right_hand: Optional[List[Landmark]] = Field(default=None, description="List of 21 right hand landmarks")

class PredictRequest(BaseModel):
    sequence: Optional[List[FrameLandmarks]] = Field(default=None)
    features: Optional[List[List[float]]] = Field(default=None)
    threshold: float = Field(default=0.70)

class TopPrediction(BaseModel):
    word: str
    probability: float

class PredictResponse(BaseModel):
    prediction: str
    probability: float
    top_predictions: List[TopPrediction]
    sequence_length: int

# --- Helper function for REST landmark extraction ---
def extract_request_landmarks(frame: FrameLandmarks) -> np.ndarray:
    pose_raw = frame.pose
    lh_raw = frame.left_hand
    rh_raw = frame.right_hand

    # Extract pose (5 landmarks)
    pose = []
    if pose_raw and len(pose_raw) == 5:
        for lm in pose_raw:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0] * 15

    # Extract left hand (21 landmarks)
    lh = []
    if lh_raw and len(lh_raw) == 21:
        for lm in lh_raw:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0] * 63

    # Extract right hand (21 landmarks)
    rh = []
    if rh_raw and len(rh_raw) == 21:
        for lm in rh_raw:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0] * 63

    vec = np.array(pose + lh + rh, dtype=np.float32)
    return predictor.normalize_frame(vec)

# --- Routes ---
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict", response_model=PredictResponse)
async def predict_api(request: PredictRequest):
    features_list = []
    
    if request.features is not None:
        features_list = request.features
        if len(features_list) == 0:
            raise HTTPException(status_code=400, detail="Features list must not be empty.")
        if any(len(frame) != 141 for frame in features_list):
            raise HTTPException(status_code=400, detail="Each feature frame must contain exactly 141 elements.")
            
    elif request.sequence is not None:
        if len(request.sequence) == 0:
            raise HTTPException(status_code=400, detail="Sequence list must not be empty.")
        for frame in request.sequence:
            features_list.append(extract_request_landmarks(frame).tolist())
            
    else:
        raise HTTPException(
            status_code=400, 
            detail="Either 'sequence' or 'features' must be provided in the request body."
        )

    # Process using the pipeline's rules
    processed_seq = predictor.process_sequence(features_list, target_length=90)
    input_data = np.expand_dims(processed_seq, axis=0)
    
    # Run prediction
    pred_probs = predictor.model.predict(input_data, verbose=0)[0]
    top_indices = np.argsort(pred_probs)[::-1]
    
    top_predictions = [
        TopPrediction(word=predictor.classes[idx], probability=float(pred_probs[idx]))
        for idx in top_indices[:3]
    ]
    
    best_idx = top_indices[0]
    
    return PredictResponse(
        prediction=predictor.classes[best_idx],
        probability=float(pred_probs[best_idx]),
        top_predictions=top_predictions,
        sequence_length=len(features_list)
    )

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    text_builder = TextBuilder()
    
    # State tracking variables per socket connection
    landmark_buffer = []
    no_hand_counter = 0
    last_prediction = ""
    prediction_cooldown = 0

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ---------- Frame message ----------
            if msg.get("type") == "frame":
                frame_b64 = msg["frame"]
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]

                img_bytes = base64.b64decode(frame_b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                cam = msg.get("camera", {})
                meta = CameraMetadata(
                    facing=cam.get("facing", "user"),
                    rotation=cam.get("rotation", 0),
                    mirrored=cam.get("mirrored", False),
                    width=cam.get("width", 640),
                    height=cam.get("height", 480),
                )
                
                threshold = msg.get("threshold", 0.70)

                # Extract and normalize frame features using pipeline
                features, hands_detected = predictor.extract_features(frame, meta)

                if hands_detected:
                    no_hand_counter = 0
                    status = "Signing Detected"
                else:
                    no_hand_counter += 1
                    if no_hand_counter > 15:
                        status = "Ready - No Signer Detected"
                        if len(landmark_buffer) > 0:
                            landmark_buffer = []
                            last_prediction = ""
                            await ws.send_text(json.dumps({
                                "type": "prediction",
                                "letter": "",
                                "confidence": 0.0,
                                "hand_detected": False,
                                "word": "",
                                "sentence": text_builder.get_full_text(),
                                "timestamp": time.time(),
                            }))
                            continue

                if hands_detected or len(landmark_buffer) > 0:
                    landmark_buffer.append(features)
                    if len(landmark_buffer) > 90:
                        landmark_buffer.pop(0)

                    # When the buffer reaches 90 frames, perform sliding window prediction
                    if len(landmark_buffer) == 90:
                        if prediction_cooldown > 0:
                            prediction_cooldown -= 1
                            # Send buffering status
                            await ws.send_text(json.dumps({
                                "type": "prediction",
                                "letter": last_prediction,
                                "confidence": 1.0,
                                "hand_detected": hands_detected,
                                "word": last_prediction,
                                "sentence": text_builder.get_full_text(),
                                "timestamp": time.time(),
                            }))
                        else:
                            # Run prediction on sequence buffer
                            pred_probs = predictor.predict_sequence(landmark_buffer)
                            top_indices = np.argsort(pred_probs)[::-1]
                            
                            best_idx = top_indices[0]
                            best_prob = float(pred_probs[best_idx])
                            best_word = predictor.classes[best_idx]

                            if best_prob >= threshold:
                                if best_word != last_prediction:
                                    # Append the newly detected word
                                    text_builder.add_word(best_word)
                                    last_prediction = best_word
                                    prediction_cooldown = 15  # Cooldown frames

                            await ws.send_text(json.dumps({
                                "type": "prediction",
                                "letter": best_word,  # display prediction overlay
                                "confidence": best_prob,
                                "hand_detected": hands_detected,
                                "word": best_word,
                                "sentence": text_builder.get_full_text(),
                                "timestamp": time.time(),
                            }))
                    else:
                        # Buffering frames
                        await ws.send_text(json.dumps({
                            "type": "prediction",
                            "letter": "",
                            "confidence": 0.0,
                            "hand_detected": hands_detected,
                            "word": f"Buffering... ({len(landmark_buffer)}/90)",
                            "sentence": text_builder.get_full_text(),
                            "timestamp": time.time(),
                        }))

            # ---------- Command message ----------
            elif msg.get("type") == "command":
                cmd = msg.get("command")
                if cmd == "clear":
                    text_builder.clear()
                elif cmd == "delete_letter":
                    text_builder.delete_letter()
                elif cmd == "delete_word":
                    text_builder.delete_word()
                elif cmd == "speak":
                    text_to_speak = text_builder.get_full_text()
                    audio_b64 = None
                    if synthesizer and text_to_speak.strip():
                        try:
                            # Run synthesis in executor to avoid blocking the event loop
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                lambda: synthesizer.speak_text_async(text_to_speak).get()
                            )
                            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                                audio_b64 = base64.b64encode(result.audio_data).decode("utf-8")
                            else:
                                print(f"Azure Speech synthesis failed: {result.reason}")
                        except Exception as ex:
                            print(f"Azure Speech error: {ex}")

                    await ws.send_text(json.dumps({
                        "type": "speak",
                        "text": text_to_speak,
                        "audio": audio_b64
                    }))
                    text_builder.clear()

                await ws.send_text(json.dumps({
                    "type": "state_update",
                    "word": "",
                    "sentence": text_builder.get_full_text(),
                }))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WS error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
