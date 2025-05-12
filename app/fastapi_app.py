from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
from models.detect_emotions import SalesCallAnalyzer, process_video
from config import MODEL_PATH, MODEL_TYPE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sales Call Emotion Analyzer API")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze a video file for emotions.

    Returns:
        FileResponse: The processed video file with emotion analysis
    """
    try:
        logger.info(f"Received video file: {file.filename}")

        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        logger.info(f"Saving temporary file to: {temp_path}")

        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Verify the video file
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        cap.release()

        # Process the video
        logger.info("Starting video processing")
        output_path = process_video(temp_path)

        # Verify the output file exists
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to generate output video")

        # Clean up temporary file
        os.remove(temp_path)
        logger.info("Temporary file removed")

        # Return the processed video
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"analyzed_{file.filename}"
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
