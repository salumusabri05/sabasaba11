# MAKISATU — Tanzanian Sign Language (TSL) Recognition API

MAKISATU (Bridging Silence) is a real-time machine learning backend built with **FastAPI**, **MediaPipe Holistic**, **Keras 3 (JAX Backend)**, and **Azure Speech SDK**. It processes high-frequency video streams and feature sequences to recognize Tanzanian Sign Language (TSL) gestures and convert them into Swahili text and spoken audio.

---

## Technical Overview

- **Real-Time Communication**: Bidirectional WebSockets (`/ws`) for 10 FPS camera frame streaming and live sentence building.
- **REST Inference**: Stateless HTTP POST endpoint (`/predict`) for landmark sequence classification.
- **Feature Extraction**: MediaPipe Holistic extracting 141-dimensional feature vectors (5 pose points + 21 left hand points + 21 right hand points).
- **ML Model**: Temporal sequence classification trained on 11 Swahili sign vocabulary classes using Keras 3 with JAX CPU acceleration.
- **Text-to-Speech**: Azure Cognitive Services Speech SDK (`sw-KE-ZuriNeural`) synthesizes accumulated text into Swahili speech audio.

---

## Project Structure

```
MAKISATU/
├── server.py               # FastAPI application & WebSocket/REST endpoints
├── pipeline.py             # Preprocessing, MediaPipe feature extraction & Keras model pipeline
├── model/
│   ├── tsl_word_model_11.h5# Trained Keras sequence model (90x141 shape)
│   └── mlp_tsl_static.pkl  # Static fallback classifier
├── static/
│   └── index.html          # Web test interface
├── docs/
│   └── index.html          # Enterprise Technical Documentation
├── Dockerfile              # Container image specification
├── FRONTEND_INTEGRATION.md # Frontend client integration guide
├── .gitlab-ci.yml          # GitLab CI/CD deployment pipeline for VPS
├── requirements.txt        # Python dependency manifest
└── .env.example            # Environment variable template
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10
- System dependencies (for OpenCV and MediaPipe):
  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0 libxcb1 libx11-xcb1 libxrender1 libxext6
  ```

### 2. Installation
```bash
# Clone the repository
git clone git@gitlab.com:salumusabri05/MAKISATU.git
cd MAKISATU

# Create & activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
PORT=5004
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=eastus
```

### 4. Running the Server
```bash
# Direct python execution
python server.py

# Or via Uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 5004 --reload
```
Open `http://localhost:5004` in a web browser to launch the live testing client.

---

## API Summary

| Type | Endpoint | Description |
| :--- | :--- | :--- |
| **HTTP GET** | `/` | Serves the web-based live camera test application |
| **WebSocket** | `/ws` | Bidirectional stream for Base64 JPEG frames & control commands |
| **HTTP POST** | `/predict` | Accepts landmark sequences `(N, 141)` and returns top-3 predictions |

---

## Deployment Specs

- **Production Server**: Contabo VPS (`169.58.79.218`), path `/var/www/makisatu`
- **Domain**: `model.bridgingsilence.org`
- **Process Manager**: PM2 (`makisatu-api`)
  ```bash
  pm2 start venv/bin/python3 --name "makisatu-api" --interpreter none -- -m uvicorn server:app --host 0.0.0.0 --port 5004
  ```
- **CI/CD**: Automated via `.gitlab-ci.yml` upon pushing to `master` branch.

---

## Technical Documentation

Full technical specifications, vector layouts, WebSocket protocol definitions, and Nginx proxy configs are detailed in the offline documentation page at:
📁 **[docs/index.html](file:///d:/BRIDGING%20SILENCE/BRIDGIND%20SILENCE%20WEBSOCKET/MAKISATU/docs/index.html)**
