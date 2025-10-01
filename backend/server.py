from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import asyncio
import cv2
import numpy as np
import torch
import json
import io
import base64
from PIL import Image
import tempfile
import shutil
import sys
from pathlib import Path
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Add python_modules to path for advanced Python features
sys.path.append('/app/python_modules')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Autism Child Activity Recognition API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for video processing
current_video_stream = None
processing_active = False

# Pydantic Models
class ActivityDetection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activity_type: str
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None
    description: str
    ai_analysis: Optional[str] = None
    video_source: str  # "live" or "uploaded"
    session_id: str

class ActivityDetectionCreate(BaseModel):
    activity_type: str
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None
    description: str
    video_source: str
    session_id: str

class VideoAnalysisRequest(BaseModel):
    video_data: str  # base64 encoded video
    session_id: str

class AnalyticsQuery(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    activity_types: Optional[List[str]] = None
    session_id: Optional[str] = None

class LabelingData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    video_frame: str  # base64 encoded frame
    labels: List[Dict[str, Any]]
    annotator_id: str
    session_id: str

class LabelingDataCreate(BaseModel):
    video_frame: str
    labels: List[Dict[str, Any]]
    annotator_id: str
    session_id: str

# Activity Recognition Engine
class ActivityRecognitionEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = 0.5
        self.activity_classes = [
            "sitting", "standing", "walking", "running", "jumping",
            "hand_flapping", "rocking", "spinning", "head_banging",
            "wandering", "aggressive_behavior", "self_harm",
            "focused_activity", "social_interaction", "therapy_exercise"
        ]
        
    def detect_person(self, frame):
        """Basic person detection using OpenCV (placeholder for Detectron2)"""
        # This is a simplified version - in production, you'd use Detectron2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'x': float(x),
                'y': float(y),
                'width': float(w),
                'height': float(h),
                'confidence': 0.8
            })
        return detections
    
    def analyze_activity(self, frame, detections):
        """Analyze detected persons for activities"""
        # Simplified activity detection - in production, use LSTM model
        activities = []
        
        for detection in detections:
            # Mock activity classification based on detection area
            area = detection['width'] * detection['height']
            if area > 10000:
                activity = "standing"
                confidence = 0.75
            elif area > 5000:
                activity = "sitting"
                confidence = 0.70
            else:
                activity = "focused_activity"
                confidence = 0.65
                
            activities.append({
                'activity_type': activity,
                'confidence': confidence,
                'bounding_box': detection,
                'description': f"Detected {activity} with {confidence:.2f} confidence"
            })
        
        return activities

# Initialize the recognition engine
recognition_engine = ActivityRecognitionEngine()

# OpenAI Integration for AI Analysis
async def get_ai_analysis(activity_data: Dict[str, Any]) -> str:
    """Get AI analysis of detected activities"""
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"analysis_{activity_data.get('session_id', 'default')}",
            system_message="""You are an expert in autism behavioral analysis. Analyze the detected activities and provide insights about:
1. Behavioral patterns
2. Potential interventions needed
3. Safety concerns if any
4. Therapeutic recommendations
Keep responses concise and professional."""
        ).with_model("openai", "gpt-5")
        
        user_message = UserMessage(
            text=f"""Analyze this activity detection:
Activity: {activity_data.get('activity_type')}
Confidence: {activity_data.get('confidence')}
Description: {activity_data.get('description')}
Timestamp: {activity_data.get('timestamp')}

Provide behavioral insights and recommendations."""
        )
        
        response = await chat.send_message(user_message)
        return response
    except Exception as e:
        logging.error(f"AI analysis error: {e}")
        return "AI analysis unavailable"

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Autism Child Activity Recognition API"}

@api_router.post("/activities", response_model=ActivityDetection)
async def create_activity_detection(input: ActivityDetectionCreate, background_tasks: BackgroundTasks):
    """Store detected activity"""
    activity_dict = input.dict()
    activity_obj = ActivityDetection(**activity_dict)
    
    # Add AI analysis in background
    if activity_obj.activity_type in ["aggressive_behavior", "self_harm", "wandering"]:
        ai_analysis = await get_ai_analysis(activity_dict)
        activity_obj.ai_analysis = ai_analysis
    
    await db.activities.insert_one(activity_obj.dict())
    return activity_obj

@api_router.get("/activities", response_model=List[ActivityDetection])
async def get_activities(session_id: Optional[str] = None, limit: int = 100):
    """Get activity detections"""
    query = {}
    if session_id:
        query["session_id"] = session_id
        
    activities = await db.activities.find(query).sort("timestamp", -1).limit(limit).to_list(limit)
    return [ActivityDetection(**activity) for activity in activities]

@api_router.post("/analyze_frame")
async def analyze_video_frame(video_data: dict):
    """Analyze a single video frame"""
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(video_data['frame'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect persons and activities
        detections = recognition_engine.detect_person(frame)
        activities = recognition_engine.analyze_activity(frame, detections)
        
        # Store significant activities
        for activity in activities:
            if activity['confidence'] > 0.6:
                activity_data = ActivityDetectionCreate(
                    **activity,
                    video_source="live",
                    session_id=video_data.get('session_id', str(uuid.uuid4()))
                )
                await create_activity_detection(activity_data, BackgroundTasks())
        
        return {
            "status": "success",
            "detections": len(detections),
            "activities": activities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame analysis failed: {str(e)}")

@api_router.post("/upload_video")
async def upload_video_analysis(file: UploadFile = File(...), session_id: str = ""):
    """Analyze uploaded video file"""
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            video_path = tmp_file.name
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_activities = []
        
        while cap.read()[0] and frame_count < 100:  # Limit processing
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 10 == 0:  # Process every 10th frame
                detections = recognition_engine.detect_person(frame)
                activities = recognition_engine.analyze_activity(frame, detections)
                
                for activity in activities:
                    if activity['confidence'] > 0.6:
                        activity_data = ActivityDetectionCreate(
                            **activity,
                            video_source="uploaded",
                            session_id=session_id or str(uuid.uuid4())
                        )
                        await create_activity_detection(activity_data, BackgroundTasks())
                        total_activities.append(activity)
            
            frame_count += 1
        
        cap.release()
        os.unlink(video_path)  # Clean up temp file
        
        return {
            "status": "success",
            "processed_frames": frame_count,
            "detected_activities": len(total_activities),
            "activities": total_activities[:10]  # Return first 10 activities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

@api_router.get("/analytics/summary")
async def get_analytics_summary(session_id: Optional[str] = None, days: int = 7):
    """Get activity analytics summary"""
    from datetime import timedelta
    
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    query = {"timestamp": {"$gte": start_date}}
    if session_id:
        query["session_id"] = session_id
    
    activities = await db.activities.find(query).to_list(1000)
    
    # Aggregate statistics
    activity_counts = {}
    hourly_distribution = [0] * 24
    confidence_sum = 0
    
    for activity in activities:
        act_type = activity['activity_type']
        activity_counts[act_type] = activity_counts.get(act_type, 0) + 1
        
        hour = activity['timestamp'].hour
        hourly_distribution[hour] += 1
        confidence_sum += activity.get('confidence', 0)
    
    avg_confidence = confidence_sum / len(activities) if activities else 0
    
    return {
        "total_activities": len(activities),
        "activity_distribution": activity_counts,
        "hourly_distribution": hourly_distribution,
        "average_confidence": avg_confidence,
        "most_common_activity": max(activity_counts.items(), key=lambda x: x[1])[0] if activity_counts else None
    }

@api_router.get("/analytics/timeline")
async def get_activity_timeline(session_id: Optional[str] = None, hours: int = 24):
    """Get activity timeline for visualization"""
    from datetime import timedelta
    
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = {"timestamp": {"$gte": start_time}}
    if session_id:
        query["session_id"] = session_id
    
    activities = await db.activities.find(query).sort("timestamp", 1).to_list(1000)
    
    timeline = []
    for activity in activities:
        timeline.append({
            "timestamp": activity["timestamp"].isoformat(),
            "activity_type": activity["activity_type"],
            "confidence": activity["confidence"],
            "description": activity["description"]
        })
    
    return {"timeline": timeline}

@api_router.post("/labeling", response_model=LabelingData)
async def create_labeling_data(input: LabelingDataCreate):
    """Store labeled training data"""
    labeling_dict = input.dict()
    labeling_obj = LabelingData(**labeling_dict)
    await db.labeling_data.insert_one(labeling_obj.dict())
    return labeling_obj

@api_router.get("/labeling", response_model=List[LabelingData])
async def get_labeling_data(limit: int = 50):
    """Get labeled data for training"""
    labeling_data = await db.labeling_data.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return [LabelingData(**data) for data in labeling_data]

@api_router.get("/alerts")
async def get_alerts(session_id: Optional[str] = None):
    """Get recent alerts for concerning activities"""
    concerning_activities = ["aggressive_behavior", "self_harm", "wandering"]
    query = {
        "activity_type": {"$in": concerning_activities},
        "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(hours=1)}
    }
    if session_id:
        query["session_id"] = session_id
    
    alerts = await db.activities.find(query).sort("timestamp", -1).to_list(10)
    return {
        "alerts": alerts,
        "count": len(alerts)
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()