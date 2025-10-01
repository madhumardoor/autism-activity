# Autism Child Activity Recognition System

## Overview

This is a comprehensive AI-powered system for monitoring and analyzing autism child activities using advanced computer vision and machine learning techniques. The system combines **Detectron2** for object detection, **LSTM models** for sequence analysis, and **OpenAI GPT-5** for intelligent activity insights.

## Architecture

### Backend (FastAPI + Python)
- **Computer Vision Pipeline**: OpenCV + PyTorch for video processing
- **Activity Detection**: Detectron2-based person detection with LSTM sequence analysis
- **AI Analysis**: OpenAI GPT-5 integration for behavioral insights
- **Data Storage**: MongoDB for activity logs, analytics, and training data
- **Real-time Processing**: Live camera feed analysis and video upload support

### Frontend (React + TypeScript)
- **Live Monitoring**: Real-time camera feed with activity detection overlay
- **Analytics Dashboard**: Historical data visualization and activity reports  
- **Video Upload**: Batch analysis of uploaded video files
- **Data Labeling**: Interface for creating training data
- **Alert System**: Real-time notifications for concerning behaviors

## Key Features

### 1. Activity Recognition
- **Basic Movements**: sitting, standing, walking, running, jumping
- **Behavioral Patterns**: hand flapping, rocking, spinning, head banging
- **Safety Monitoring**: wandering detection, aggressive behavior, self-harm indicators
- **Therapeutic Activities**: focused activities, social interactions, therapy exercises

### 2. Monitoring Capabilities
- ✅ **Live Camera Feed**: Real-time video stream analysis
- ✅ **Video File Upload**: Batch processing of recorded videos
- ✅ **Activity Timeline**: Historical activity tracking and visualization
- ✅ **Confidence Scoring**: Machine learning confidence metrics for all detections

### 3. AI-Powered Insights
- **GPT-5 Analysis**: Advanced behavioral pattern analysis and recommendations
- **Safety Alerts**: Automatic detection of concerning activities
- **Therapeutic Insights**: Personalized intervention suggestions
- **Progress Tracking**: Long-term behavioral trend analysis

### 4. Data Management
- **Training Data Collection**: Built-in labeling interface for model improvement
- **Analytics Dashboard**: Comprehensive activity statistics and reports
- **Export Capabilities**: Data export for external analysis
- **Privacy Controls**: Secure data handling and storage

## Technology Stack

### ML/AI Components
- **Detectron2**: Facebook's state-of-the-art object detection framework
- **PyTorch**: Deep learning framework for LSTM sequence modeling
- **OpenCV**: Computer vision library for video processing
- **OpenAI GPT-5**: Large language model for activity analysis
- **scikit-learn**: Additional ML utilities and preprocessing

### Backend
- **FastAPI**: High-performance Python web framework
- **MongoDB**: Document database for flexible data storage
- **Motor**: Async MongoDB driver
- **Emergent Integrations**: Universal LLM integration library

### Frontend
- **React 19**: Modern UI framework
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/UI**: Premium component library
- **Axios**: HTTP client for API communication
- **React Router**: Single-page application routing

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB
- Camera access (for live monitoring)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
yarn install
```

### Environment Configuration
- **Backend**: `/app/backend/.env`
- **Frontend**: `/app/frontend/.env`

## API Endpoints

### Activity Detection
- `POST /api/analyze_frame` - Analyze single video frame
- `POST /api/upload_video` - Upload and analyze video file
- `GET /api/activities` - Retrieve activity detections
- `POST /api/activities` - Store new activity detection

### Analytics & Monitoring  
- `GET /api/analytics/summary` - Activity statistics summary
- `GET /api/analytics/timeline` - Activity timeline data
- `GET /api/alerts` - Recent safety alerts

### Data Management
- `POST /api/labeling` - Store labeled training data
- `GET /api/labeling` - Retrieve labeled data

## Usage Guide

### 1. Live Monitoring
1. Navigate to the "Monitor" tab
2. Click "Start Camera" to begin live video feed
3. System automatically detects and classifies activities
4. View real-time activity feed and alerts in dashboard

### 2. Video Analysis
1. Go to "Upload" tab  
2. Select video file (MP4, AVI, MOV supported)
3. System processes video and extracts activity data
4. Results appear in analytics dashboard

### 3. Data Labeling
1. Navigate to "Labeling" tab
2. Upload images for manual labeling
3. Select relevant activity types
4. Save labeled data for model training

### 4. Analytics Review
1. Visit "Analytics" tab for comprehensive reports
2. View activity distribution, confidence metrics
3. Monitor hourly activity patterns
4. Review AI-generated behavioral insights

## Activity Categories

### Movement Activities
- **sitting**: Child in seated position
- **standing**: Upright stationary position
- **walking**: Normal locomotion
- **running**: Fast locomotion
- **jumping**: Vertical movement patterns

### Behavioral Indicators
- **hand_flapping**: Repetitive hand movements (stimming)
- **rocking**: Back-and-forth body movement
- **spinning**: Rotational body movement
- **head_banging**: Self-injurious head movements

### Safety Concerns
- **wandering**: Unstructured movement away from designated areas
- **aggressive_behavior**: Physical aggression indicators
- **self_harm**: Self-injurious behaviors

### Therapeutic Activities
- **focused_activity**: Sustained attention on tasks
- **social_interaction**: Engagement with others
- **therapy_exercise**: Structured therapeutic activities

## Machine Learning Pipeline

### 1. Person Detection (Detectron2)
```python
# Detectron2 model for person detection
model = build_model(cfg)
outputs = model(frame)
persons = outputs["instances"].pred_boxes
```

### 2. Activity Classification (LSTM)
```python
# LSTM for sequence analysis
lstm_model = torch.nn.LSTM(input_size, hidden_size, num_layers)
activity_prediction = lstm_model(sequence_features)
```

### 3. AI Analysis (GPT-5)
```python
# OpenAI GPT-5 for behavioral insights
response = await chat.send_message(activity_context)
behavioral_insights = response.content
```

## Quick Start Commands

```bash
# Start the complete system
sudo supervisorctl restart all

# Check system status
sudo supervisorctl status

# View backend logs
tail -f /var/log/supervisor/backend.*.log

# Frontend development
cd frontend && yarn start

# Backend development  
cd backend && uvicorn server:app --reload
```
