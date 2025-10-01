import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';
import { Progress } from './components/ui/progress';
import { Separator } from './components/ui/separator';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Live Camera Component
const LiveCamera = ({ onActivityDetected }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const intervalRef = useRef(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      videoRef.current.srcObject = stream;
      setIsStreaming(true);
      
      // Start frame analysis
      intervalRef.current = setInterval(analyzeFrame, 2000); // Analyze every 2 seconds
      toast.success("Camera started successfully");
    } catch (error) {
      toast.error("Failed to access camera: " + error.message);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsStreaming(false);
    toast.info("Camera stopped");
  };

  const analyzeFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    ctx.drawImage(videoRef.current, 0, 0);
    
    // Convert to base64
    const frameData = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    
    try {
      const response = await axios.post(`${API}/analyze_frame`, {
        frame: frameData,
        session_id: sessionId
      });
      
      if (response.data.activities && response.data.activities.length > 0) {
        onActivityDetected && onActivityDetected(response.data.activities);
      }
    } catch (error) {
      console.error('Frame analysis error:', error);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <Card data-testid="live-camera-card">
      <CardHeader>
        <CardTitle>Live Camera Monitor</CardTitle>
        <CardDescription>Real-time activity detection</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex gap-4">
            <Button 
              onClick={startCamera} 
              disabled={isStreaming}
              data-testid="start-camera-btn"
              className="bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700"
            >
              Start Camera
            </Button>
            <Button 
              onClick={stopCamera} 
              disabled={!isStreaming}
              data-testid="stop-camera-btn"
              variant="outline"
            >
              Stop Camera
            </Button>
          </div>
          
          <div className="relative">
            <video 
              ref={videoRef} 
              autoPlay 
              muted 
              className="w-full rounded-lg border"
              data-testid="video-stream"
            />
            <canvas ref={canvasRef} className="hidden" />
            {isStreaming && (
              <Badge className="absolute top-2 left-2 bg-red-500 animate-pulse">
                LIVE
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Video Upload Component
const VideoUpload = ({ onVideoUploaded }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', `upload_${Date.now()}`);

    try {
      const response = await axios.post(`${API}/upload_video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      toast.success(`Video analyzed! Found ${response.data.detected_activities} activities`);
      onVideoUploaded && onVideoUploaded(response.data);
    } catch (error) {
      toast.error("Video upload failed: " + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <Card data-testid="video-upload-card">
      <CardHeader>
        <CardTitle>Video Analysis</CardTitle>
        <CardDescription>Upload video files for activity analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              disabled={uploading}
              className="hidden"
              id="video-upload"
              data-testid="video-input"
            />
            <label 
              htmlFor="video-upload" 
              className="cursor-pointer"
              data-testid="upload-label"
            >
              <div className="space-y-2">
                <div className="text-4xl">📹</div>
                <p className="text-lg font-medium">
                  {uploading ? 'Analyzing...' : 'Click to upload video'}
                </p>
                <p className="text-sm text-gray-500">
                  Supports MP4, AVI, MOV files
                </p>
              </div>
            </label>
          </div>
          
          {uploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Activity Dashboard Component
const ActivityDashboard = () => {
  const [activities, setActivities] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetchActivities();
    fetchAnalytics();
    fetchAlerts();
    
    const interval = setInterval(() => {
      fetchActivities();
      fetchAlerts();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchActivities = async () => {
    try {
      const response = await axios.get(`${API}/activities?limit=10`);
      setActivities(response.data);
    } catch (error) {
      console.error('Failed to fetch activities:', error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const response = await axios.get(`${API}/analytics/summary`);
      setAnalytics(response.data);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API}/alerts`);
      setAlerts(response.data.alerts || []);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const getActivityColor = (activity) => {
    const concerningActivities = ['aggressive_behavior', 'self_harm', 'wandering'];
    if (concerningActivities.includes(activity)) return 'bg-red-500';
    return 'bg-blue-500';
  };

  return (
    <div className="space-y-6" data-testid="activity-dashboard">
      {/* Alerts Section */}
      {alerts.length > 0 && (
        <Alert className="border-red-200 bg-red-50" data-testid="alerts-section">
          <AlertTitle className="text-red-800">⚠️ Active Alerts</AlertTitle>
          <AlertDescription>
            <div className="space-y-2 mt-2">
              {alerts.map((alert, index) => (
                <div key={index} className="text-sm text-red-700">
                  <span className="font-medium">{alert.activity_type}</span> detected at{' '}
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </div>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Analytics Summary */}
      {analytics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card data-testid="total-activities-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Total Activities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{analytics.total_activities}</div>
            </CardContent>
          </Card>
          
          <Card data-testid="avg-confidence-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Avg Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(analytics.average_confidence * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>
          
          <Card data-testid="most-common-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Most Common</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-lg font-medium">{analytics.most_common_activity || 'N/A'}</div>
            </CardContent>
          </Card>
          
          <Card data-testid="activity-types-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Activity Types</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{Object.keys(analytics.activity_distribution).length}</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Recent Activities */}
      <Card data-testid="recent-activities-card">
        <CardHeader>
          <CardTitle>Recent Activities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {activities.map((activity) => (
              <div key={activity.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <Badge className={getActivityColor(activity.activity_type)}>
                    {activity.activity_type.replace('_', ' ')}
                  </Badge>
                  <span className="text-sm text-gray-600">
                    {new Date(activity.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">{(activity.confidence * 100).toFixed(1)}%</div>
                  <div className="text-xs text-gray-500">{activity.video_source}</div>
                </div>
              </div>
            ))}
            {activities.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No activities detected yet. Start monitoring to see results.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Data Labeling Component
const DataLabeling = () => {
  const [labelingData, setLabelingData] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [labels, setLabels] = useState([]);

  useEffect(() => {
    fetchLabelingData();
  }, []);

  const fetchLabelingData = async () => {
    try {
      const response = await axios.get(`${API}/labeling`);
      setLabelingData(response.data);
    } catch (error) {
      console.error('Failed to fetch labeling data:', error);
    }
  };

  const saveLabeledData = async () => {
    if (!selectedImage || labels.length === 0) return;

    try {
      await axios.post(`${API}/labeling`, {
        video_frame: selectedImage,
        labels: labels,
        annotator_id: 'user_1',
        session_id: `labeling_${Date.now()}`
      });
      
      toast.success('Labeled data saved successfully!');
      setLabels([]);
      fetchLabelingData();
    } catch (error) {
      toast.error('Failed to save labeled data');
    }
  };

  return (
    <Card data-testid="data-labeling-card">
      <CardHeader>
        <CardTitle>Data Labeling</CardTitle>
        <CardDescription>Create training data by labeling activities</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-medium mb-2">Upload Image for Labeling</h3>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files[0];
                  if (file) {
                    const reader = new FileReader();
                    reader.onload = (e) => setSelectedImage(e.target.result);
                    reader.readAsDataURL(file);
                  }
                }}
                className="w-full"
                data-testid="image-upload-input"
              />
              {selectedImage && (
                <img 
                  src={selectedImage} 
                  alt="For labeling" 
                  className="mt-2 w-full h-48 object-cover rounded border"
                />
              )}
            </div>
            
            <div>
              <h3 className="font-medium mb-2">Activity Labels</h3>
              <div className="space-y-2">
                {['sitting', 'standing', 'walking', 'hand_flapping', 'rocking'].map(activity => (
                  <label key={activity} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={labels.some(l => l.activity === activity)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setLabels([...labels, { activity, confidence: 1.0 }]);
                        } else {
                          setLabels(labels.filter(l => l.activity !== activity));
                        }
                      }}
                    />
                    <span className="capitalize">{activity.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
              <Button 
                onClick={saveLabeledData} 
                className="mt-4 w-full"
                disabled={!selectedImage || labels.length === 0}
                data-testid="save-labels-btn"
              >
                Save Labeled Data
              </Button>
            </div>
          </div>
          
          <Separator />
          
          <div>
            <h3 className="font-medium mb-2">Existing Labeled Data ({labelingData.length} items)</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {labelingData.slice(0, 8).map((item) => (
                <div key={item.id} className="border rounded p-2 text-xs">
                  <div className="font-medium">{item.labels.length} labels</div>
                  <div className="text-gray-500">{new Date(item.timestamp).toLocaleDateString()}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Main App Component
const Home = () => {
  const [currentTab, setCurrentTab] = useState('monitor');

  const handleActivityDetected = (activities) => {
    activities.forEach(activity => {
      if (activity.confidence > 0.7) {
        toast.info(`Detected: ${activity.activity_type} (${(activity.confidence * 100).toFixed(1)}%)`);
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Autism Child Activity Recognition
          </h1>
          <p className="text-lg text-gray-600">
            AI-powered monitoring and analysis system using Detectron2 + LSTM + GPT-5
          </p>
        </div>

        <Tabs value={currentTab} onValueChange={setCurrentTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4" data-testid="main-tabs">
            <TabsTrigger value="monitor" data-testid="monitor-tab">Monitor</TabsTrigger>
            <TabsTrigger value="analytics" data-testid="analytics-tab">Analytics</TabsTrigger>
            <TabsTrigger value="upload" data-testid="upload-tab">Upload</TabsTrigger>
            <TabsTrigger value="labeling" data-testid="labeling-tab">Labeling</TabsTrigger>
          </TabsList>

          <TabsContent value="monitor" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LiveCamera onActivityDetected={handleActivityDetected} />
              <ActivityDashboard />
            </div>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-6">
            <ActivityDashboard />
          </TabsContent>

          <TabsContent value="upload" className="space-y-6">
            <VideoUpload />
          </TabsContent>

          <TabsContent value="labeling" className="space-y-6">
            <DataLabeling />
          </TabsContent>
        </Tabs>
      </div>
      <Toaster />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;