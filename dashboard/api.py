"""
=========================================================
FastAPI Backend
AI-Powered Multi-Modal Cattle Health Monitoring
=========================================================

Runs the complete pipeline after a video upload.

Pipeline

1. Save uploaded video
2. Vision Pipeline
3. Sensor Generator
4. Sensor Summary
5. Risk Score
6. Decision Fusion
7. Recommendation
8. SHAP XAI

Author:
Sajid Islam
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
import subprocess
import os
import time

app = FastAPI(
    title="AI Cattle Health Monitoring API",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Directories
# ----------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VISION_INPUT = os.path.join(
    PROJECT_ROOT,
    "vision",
    "input"
)

OUTPUT = os.path.join(
    PROJECT_ROOT,
    "generated"
)

os.makedirs(VISION_INPUT, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ----------------------------------------------------
# Helper
# ----------------------------------------------------

def run_python(script):

    script_path = os.path.join(PROJECT_ROOT, script)

    start = time.time()

    result = subprocess.run(

        ["python3", script_path],

        cwd=PROJECT_ROOT,

        capture_output=True,

        text=True

    )

    runtime = round(time.time() - start, 2)

    if result.returncode != 0:

        raise RuntimeError(result.stderr)

    return runtime


# ----------------------------------------------------
# Health Check
# ----------------------------------------------------

@app.get("/")

def root():

    return {

        "status": "running",

        "project": "AI Cattle Health Monitoring"

    }


# ----------------------------------------------------
# Run Whole Pipeline
# ----------------------------------------------------

@app.post("/run")

async def run_pipeline(video: UploadFile = File(...)):

    try:

        # --------------------------------------------
        # Save Uploaded Video
        # --------------------------------------------

        save_path = os.path.join(
            VISION_INPUT,
            "cow.mp4"
        )

        with open(save_path, "wb") as buffer:

            shutil.copyfileobj(video.file, buffer)

        logs = []

        # --------------------------------------------
        # Vision
        # --------------------------------------------

        t = run_python("vision/run_pipeline.py")

        logs.append({
            "step": "Vision Pipeline",
            "time": t
        })

        # --------------------------------------------
        # Synthetic Sensor
        # --------------------------------------------

        t = run_python("sensor_generator.py")

        logs.append({
            "step": "Sensor Generator",
            "time": t
        })

        # --------------------------------------------
        # Sensor Summary
        # --------------------------------------------

        t = run_python("sensor_summary.py")

        logs.append({
            "step": "Sensor Summary",
            "time": t
        })

        # --------------------------------------------
        # Risk Score
        # --------------------------------------------

        t = run_python("risk_score_engine.py")

        logs.append({
            "step": "Risk Score",
            "time": t
        })

        # --------------------------------------------
        # Decision Fusion
        # --------------------------------------------

        t = run_python("decision_fusion_engine.py")

        logs.append({
            "step": "Decision Fusion",
            "time": t
        })

        # --------------------------------------------
        # Recommendation
        # --------------------------------------------

        t = run_python("recommendation_engine.py")

        logs.append({
            "step": "Recommendation",
            "time": t
        })

        # --------------------------------------------
        # SHAP Explainability
        # --------------------------------------------

        t = run_python("shap_xai_engine.py")

        logs.append({
            "step": "SHAP XAI",
            "time": t
        })

        return JSONResponse({

            "status": "success",

            "message": "Pipeline Completed",

            "result_csv": "generated/final_prediction.csv",

            "logs": logs

        })

    except Exception as e:

        import traceback

        return JSONResponse(

            status_code=500,

            content={

                "status": "error",

                "message": str(e),

                "traceback": traceback.format_exc()

            }

        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)