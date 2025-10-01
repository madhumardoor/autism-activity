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

# Advanced Python Integration Endpoints
@api_router.post("/python/data_analysis")
async def run_data_analysis():
    """
    Advanced Python data analysis endpoint
    Demonstrates: Import of custom Python modules, async operations
    """
    try:
        # Import custom data analysis module
        from data_analysis import AutismDataAnalyzer
        
        # Initialize analyzer
        mongo_url = os.environ['MONGO_URL']
        async with AutismDataAnalyzer(mongo_url) as analyzer:
            
            # Fetch recent data
            df = await analyzer.fetch_activities_data(days_back=7)
            
            if df.empty:
                return {"message": "No data available for analysis", "patterns": []}
            
            # Generate patterns using advanced Python features
            patterns = analyzer.generate_activity_patterns(df)
            
            # Generate insights
            insights = analyzer.generate_behavioral_insights(patterns)
            
            return {
                "status": "success",
                "data_points": len(df),
                "patterns_found": len(patterns),
                "insights": insights,
                "top_patterns": [
                    {
                        "activity": p.activity_type,
                        "frequency": p.frequency,
                        "confidence": round(p.avg_confidence, 3),
                        "category": p.behavioral_category
                    } for p in patterns[:5]
                ]
            }
    
    except ImportError as e:
        return {"error": f"Python module import failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Data analysis failed: {str(e)}"}

@api_router.post("/python/ml_prediction")
async def run_ml_prediction():
    """
    Machine learning prediction endpoint
    Demonstrates: Advanced ML with PyTorch/scikit-learn integration
    """
    try:
        from ml_models import RandomForestActivityClassifier, FeatureExtractor
        import numpy as np
        
        # Generate sample features (in real scenario, this would come from video frames)
        extractor = FeatureExtractor()
        sample_sequence = np.random.randn(100)  # Simulated activity sequence
        features = extractor.extract_statistical_features(sample_sequence)
        
        # For demo purposes, create sample training data
        np.random.seed(42)
        X_train = np.random.randn(100, len(features))
        y_train = np.random.choice(['sitting', 'walking', 'hand_flapping', 'focused_activity'], 100)
        
        # Train and predict using Random Forest
        rf_model = RandomForestActivityClassifier(n_estimators=50)
        rf_model.train(X_train, y_train)
        
        # Make prediction on extracted features
        prediction = rf_model.predict(features.reshape(1, -1))
        
        # Get feature importance
        feature_importance = rf_model.get_feature_importance()
        
        return {
            "status": "success",
            "prediction": prediction[0],
            "confidence": 0.85,  # Simulated confidence
            "features_extracted": len(features),
            "model_type": "RandomForest",
            "feature_importance": dict(list(feature_importance.items())[:5])
        }
    
    except Exception as e:
        return {"error": f"ML prediction failed: {str(e)}"}

@api_router.get("/python/system_metrics")
async def get_system_metrics():
    """
    System monitoring endpoint using Python automation
    Demonstrates: System monitoring, performance metrics
    """
    try:
        from automation_scripts import SystemMonitor, AutomationConfig
        
        config = AutomationConfig()
        monitor = SystemMonitor(config)
        
        # Get current system metrics
        metrics = monitor.get_system_metrics()
        
        # Get metrics summary
        summary = monitor.get_metrics_summary(hours=1)
        
        return {
            "status": "success",
            "current_metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage_percent": metrics.disk_usage_percent,
                "process_count": metrics.process_count,
                "timestamp": metrics.timestamp.isoformat()
            },
            "summary": summary,
            "python_features": [
                "Advanced system monitoring",
                "Performance metrics collection", 
                "Automated alerting system",
                "Task scheduling capabilities"
            ]
        }
    
    except Exception as e:
        return {"error": f"System monitoring failed: {str(e)}"}

@api_router.post("/python/generate_visualization")
async def generate_visualization():
    """
    Data visualization generation endpoint
    Demonstrates: Matplotlib, Seaborn, Plotly integration
    """
    try:
        from data_visualization import ActivityVisualizationEngine
        
        mongo_url = os.environ['MONGO_URL']
        output_dir = "/app/visualizations"
        
        # Initialize visualization engine
        viz_engine = ActivityVisualizationEngine(mongo_url, output_dir)
        
        # Generate visualization report
        report_paths = await viz_engine.generate_comprehensive_report(days_back=7)
        
        if 'error' in report_paths:
            return {
                "status": "limited",
                "message": "Limited visualizations due to insufficient data",
                "error": report_paths['error']
            }
        
        # Convert absolute paths to relative for API response
        relative_paths = {}
        for viz_type, path in report_paths.items():
            if viz_type != 'error':
                relative_paths[viz_type] = Path(path).name
        
        return {
            "status": "success",
            "visualizations_generated": len(relative_paths),
            "available_visualizations": list(relative_paths.keys()),
            "output_directory": output_dir,
            "python_libraries": [
                "Matplotlib for static plots",
                "Seaborn for statistical visualization",
                "Plotly for interactive dashboards",
                "Pandas for data manipulation"
            ]
        }
    
    except Exception as e:
        return {"error": f"Visualization generation failed: {str(e)}"}

@api_router.post("/python/run_demo")
async def run_python_demo():
    """
    Comprehensive Python demonstration endpoint
    Showcases all advanced Python features in one endpoint
    """
    try:
        # Create demo results
        demo_results = {}
        
        # 1. Advanced Python Features Demo
        def demonstrate_python_features():
            # Generators
            def fibonacci(n):
                a, b = 0, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b
            
            fib_sequence = list(fibonacci(10))
            
            # List comprehensions
            squares = [x**2 for x in range(1, 11) if x % 2 == 0]
            
            # Dictionary comprehensions  
            square_dict = {x: x**2 for x in range(1, 6)}
            
            # Lambda functions
            numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            evens = list(filter(lambda x: x % 2 == 0, numbers))
            
            return {
                'fibonacci_sequence': fib_sequence,
                'even_squares': squares,
                'square_dictionary': square_dict,
                'filtered_evens': evens
            }
        
        # 2. Async/Await Patterns
        async def async_computation():
            await asyncio.sleep(0.1)  # Simulate async work
            return "Async computation completed"
        
        # 3. Context Managers
        class TimingContext:
            def __enter__(self):
                self.start = datetime.now()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.duration = (datetime.now() - self.start).total_seconds()
        
        # 4. Decorators
        def timing_decorator(func):
            def wrapper(*args, **kwargs):
                start = datetime.now()
                result = func(*args, **kwargs)
                duration = (datetime.now() - start).total_seconds()
                return {'result': result, 'execution_time': duration}
            return wrapper
        
        @timing_decorator
        def sample_computation():
            return sum(range(1000))
        
        # Execute demonstrations
        python_features = demonstrate_python_features()
        async_result = await async_computation()
        
        with TimingContext() as timer:
            computation_result = sample_computation()
        
        # Compile results
        demo_results = {
            "status": "success",
            "python_version": sys.version,
            "features_demonstrated": [
                "Generators and yield statements",
                "List and dictionary comprehensions", 
                "Lambda functions and functional programming",
                "Async/await patterns",
                "Context managers",
                "Decorators and function wrapping",
                "Advanced data structures",
                "Type hints and dataclasses"
            ],
            "results": {
                "advanced_features": python_features,
                "async_computation": async_result,
                "context_manager_timing": timer.duration,
                "decorated_computation": computation_result
            },
            "modules_imported": [
                "data_analysis - Advanced pandas/numpy operations",
                "ml_models - PyTorch/scikit-learn integration", 
                "automation_scripts - System monitoring and task scheduling",
                "data_visualization - Matplotlib/Seaborn/Plotly charts"
            ]
        }
        
        return demo_results
    
    except Exception as e:
        return {"error": f"Python demo failed: {str(e)}"}

@api_router.get("/python/capabilities")
async def get_python_capabilities():
    """
    Get overview of all Python capabilities implemented
    """
    return {
        "status": "success",
        "python_version": sys.version,
        "advanced_features": {
            "data_analysis": {
                "description": "Advanced pandas/numpy data processing",
                "features": [
                    "Async database operations",
                    "Statistical analysis and clustering",
                    "Behavioral pattern recognition",
                    "Data processing pipelines",
                    "Generator functions for memory efficiency"
                ]
            },
            "machine_learning": {
                "description": "PyTorch and scikit-learn ML models",
                "features": [
                    "LSTM neural networks for sequence analysis",
                    "Random Forest and SVM classifiers",
                    "Feature extraction and engineering",
                    "Model evaluation and comparison",
                    "Custom dataset classes and training loops"
                ]
            },
            "automation": {
                "description": "System automation and monitoring",
                "features": [
                    "Task scheduling with decorators",
                    "System performance monitoring",
                    "Database backup and maintenance",
                    "Email and webhook notifications",
                    "Async context managers"
                ]
            },
            "visualization": {
                "description": "Advanced data visualization",
                "features": [
                    "Statistical plots with matplotlib/seaborn",
                    "Interactive dashboards with Plotly", 
                    "Heatmaps and correlation analysis",
                    "Time series visualization",
                    "Machine learning result visualization"
                ]
            },
            "programming_concepts": {
                "description": "Advanced Python programming patterns",
                "features": [
                    "Generators and iterators",
                    "Decorators and metaclasses",
                    "Context managers",
                    "Async/await programming",
                    "Type hints and dataclasses",
                    "Functional programming patterns",
                    "Abstract base classes and inheritance"
                ]
            }
        },
        "libraries_integrated": [
            "pandas", "numpy", "matplotlib", "seaborn", "plotly",
            "torch", "torchvision", "scikit-learn", "opencv-python",
            "motor", "asyncio", "schedule", "psutil", "requests"
        ],
        "endpoints": [
            "POST /api/python/data_analysis - Advanced data analysis",
            "POST /api/python/ml_prediction - Machine learning predictions", 
            "GET /api/python/system_metrics - System monitoring",
            "POST /api/python/generate_visualization - Data visualizations",
            "POST /api/python/run_demo - Comprehensive Python demo",
            "GET /api/python/capabilities - This endpoint"
        ]
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