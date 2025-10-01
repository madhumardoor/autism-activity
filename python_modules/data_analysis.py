"""
Advanced Python Data Analysis Module for Autism Activity Recognition
Demonstrates: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import asyncio
import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv('/app/backend/.env')

@dataclass
class ActivityPattern:
    """Data class for activity pattern analysis"""
    activity_type: str
    frequency: int
    avg_confidence: float
    duration_minutes: float
    time_of_day: int
    behavioral_category: str

class AutismDataAnalyzer:
    """
    Comprehensive Python data analysis class for autism monitoring data
    Showcases advanced Python concepts: async/await, decorators, context managers,
    generators, type hints, dataclasses, and more
    """
    
    def __init__(self, db_connection_string: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(db_connection_string)
        self.db = self.client[os.environ.get('DB_NAME', 'autism_monitoring_db')]
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.client.close()
    
    def timing_decorator(func):
        """Decorator to measure function execution time"""
        import time
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} executed in {end - start:.4f} seconds")
            return result
        return wrapper
    
    async def fetch_activities_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch activity data from MongoDB and convert to pandas DataFrame
        Demonstrates: async database operations, pandas data manipulation
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        cursor = self.db.activities.find({
            "timestamp": {"$gte": start_date}
        })
        
        activities = await cursor.to_list(length=None)
        
        if not activities:
            return pd.DataFrame()
        
        # Convert to DataFrame with advanced pandas operations
        df = pd.DataFrame(activities)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        # Feature engineering
        df['confidence_category'] = pd.cut(df['confidence'], 
                                         bins=[0, 0.7, 0.85, 1.0], 
                                         labels=['Low', 'Medium', 'High'])
        
        # Behavioral categorization
        behavioral_map = {
            'sitting': 'Calm',
            'standing': 'Calm', 
            'walking': 'Active',
            'running': 'Active',
            'jumping': 'Active',
            'hand_flapping': 'Stimming',
            'rocking': 'Stimming',
            'spinning': 'Stimming',
            'head_banging': 'Concerning',
            'wandering': 'Concerning',
            'aggressive_behavior': 'Concerning',
            'self_harm': 'Concerning',
            'focused_activity': 'Therapeutic',
            'social_interaction': 'Therapeutic',
            'therapy_exercise': 'Therapeutic'
        }
        
        df['behavioral_category'] = df['activity_type'].map(behavioral_map)
        
        return df
    
    @timing_decorator
    def generate_activity_patterns(self, df: pd.DataFrame) -> List[ActivityPattern]:
        """
        Generate activity patterns using advanced Python data structures
        Demonstrates: list comprehensions, generators, dataclasses
        """
        if df.empty:
            return []
        
        patterns = []
        
        # Group by activity type and analyze patterns
        for activity_type in df['activity_type'].unique():
            activity_data = df[df['activity_type'] == activity_type]
            
            pattern = ActivityPattern(
                activity_type=activity_type,
                frequency=len(activity_data),
                avg_confidence=activity_data['confidence'].mean(),
                duration_minutes=self._calculate_duration(activity_data),
                time_of_day=activity_data['hour'].mode().iloc[0] if not activity_data['hour'].mode().empty else 12,
                behavioral_category=activity_data['behavioral_category'].iloc[0]
            )
            patterns.append(pattern)
        
        # Sort by frequency (descending)
        return sorted(patterns, key=lambda x: x.frequency, reverse=True)
    
    def _calculate_duration(self, activity_data: pd.DataFrame) -> float:
        """
        Calculate average duration between activities
        Demonstrates: private methods, pandas datetime operations
        """
        if len(activity_data) < 2:
            return 5.0  # Default 5 minutes
        
        activity_data_sorted = activity_data.sort_values('timestamp')
        time_diffs = activity_data_sorted['timestamp'].diff().dt.total_seconds() / 60
        return time_diffs.mean()
    
    def activity_clustering_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform K-means clustering on activity patterns
        Demonstrates: scikit-learn, numpy arrays, machine learning
        """
        if df.empty:
            return {"error": "No data available for clustering"}
        
        # Prepare features for clustering
        features = df.groupby('activity_type').agg({
            'confidence': ['mean', 'std'],
            'hour': ['mean', 'std'],
            'timestamp': 'count'
        }).fillna(0)
        
        # Flatten multi-level columns
        features.columns = ['_'.join(col).strip() for col in features.columns]
        
        if len(features) < 2:
            return {"error": "Insufficient data for clustering"}
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply K-means clustering
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to features
        features['cluster'] = clusters
        
        return {
            'cluster_centers': self.kmeans.cluster_centers_.tolist(),
            'cluster_labels': clusters.tolist(),
            'activity_clusters': features['cluster'].to_dict(),
            'inertia': self.kmeans.inertia_
        }
    
    def generate_behavioral_insights(self, patterns: List[ActivityPattern]) -> Dict[str, any]:
        """
        Generate behavioral insights using Python data analysis
        Demonstrates: dictionary comprehensions, advanced data aggregation
        """
        if not patterns:
            return {"insights": [], "summary": "No data available"}
        
        # Categorize patterns by behavioral type
        behavioral_groups = {}
        for pattern in patterns:
            if pattern.behavioral_category not in behavioral_groups:
                behavioral_groups[pattern.behavioral_category] = []
            behavioral_groups[pattern.behavioral_category].append(pattern)
        
        insights = []
        
        # Generate insights for each category
        for category, category_patterns in behavioral_groups.items():
            total_frequency = sum(p.frequency for p in category_patterns)
            avg_confidence = np.mean([p.avg_confidence for p in category_patterns])
            
            insight = {
                'category': category,
                'total_occurrences': total_frequency,
                'average_confidence': round(avg_confidence, 3),
                'activity_count': len(category_patterns),
                'dominant_activities': [p.activity_type for p in category_patterns[:2]],
                'risk_level': self._assess_risk_level(category, total_frequency, avg_confidence)
            }
            insights.append(insight)
        
        # Generate summary statistics
        summary = {
            'total_patterns': len(patterns),
            'most_frequent_activity': patterns[0].activity_type if patterns else None,
            'overall_confidence': round(np.mean([p.avg_confidence for p in patterns]), 3),
            'behavioral_diversity': len(behavioral_groups),
            'concerning_behaviors': len([p for p in patterns if p.behavioral_category == 'Concerning'])
        }
        
        return {
            'insights': insights,
            'summary': summary,
            'recommendations': self._generate_recommendations(behavioral_groups)
        }
    
    def _assess_risk_level(self, category: str, frequency: int, confidence: float) -> str:
        """
        Assess risk level based on behavioral category and frequency
        Demonstrates: conditional logic, risk assessment algorithms
        """
        if category == 'Concerning':
            if frequency > 5 and confidence > 0.8:
                return 'High'
            elif frequency > 2 and confidence > 0.7:
                return 'Medium'
            else:
                return 'Low'
        elif category == 'Stimming':
            if frequency > 10:
                return 'Medium'
            else:
                return 'Low'
        else:
            return 'Low'
    
    def _generate_recommendations(self, behavioral_groups: Dict) -> List[str]:
        """
        Generate therapeutic recommendations based on behavioral patterns
        Demonstrates: complex conditional logic, therapeutic algorithms
        """
        recommendations = []
        
        if 'Concerning' in behavioral_groups:
            concerning_count = len(behavioral_groups['Concerning'])
            if concerning_count > 2:
                recommendations.append("Immediate intervention recommended for concerning behaviors")
            recommendations.append("Implement safety monitoring protocols")
        
        if 'Stimming' in behavioral_groups:
            recommendations.append("Consider sensory regulation activities")
            recommendations.append("Monitor stimming patterns for triggers")
        
        if 'Therapeutic' in behavioral_groups:
            therapeutic_count = len(behavioral_groups['Therapeutic'])
            if therapeutic_count > 0:
                recommendations.append("Continue current therapeutic activities - showing positive engagement")
            else:
                recommendations.append("Introduce more structured therapeutic activities")
        
        if 'Active' in behavioral_groups and 'Calm' in behavioral_groups:
            recommendations.append("Good balance of active and calm activities observed")
        
        return recommendations
    
    async def export_analysis_report(self, output_path: str = '/app/analysis_report.json') -> str:
        """
        Generate comprehensive analysis report and export to JSON
        Demonstrates: file I/O, JSON serialization, async operations
        """
        import json
        
        # Fetch and analyze data
        df = await self.fetch_activities_data(days_back=30)
        patterns = self.generate_activity_patterns(df)
        clustering_results = self.activity_clustering_analysis(df)
        insights = self.generate_behavioral_insights(patterns)
        
        # Generate comprehensive report
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_summary': {
                'total_activities': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if not df.empty else "No data",
                'unique_activity_types': df['activity_type'].nunique() if not df.empty else 0
            },
            'activity_patterns': [
                {
                    'activity_type': p.activity_type,
                    'frequency': p.frequency,
                    'avg_confidence': round(p.avg_confidence, 3),
                    'behavioral_category': p.behavioral_category
                } for p in patterns
            ],
            'clustering_analysis': clustering_results,
            'behavioral_insights': insights,
            'python_analysis_metadata': {
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__,
                'analysis_functions_used': [
                    'fetch_activities_data',
                    'generate_activity_patterns', 
                    'activity_clustering_analysis',
                    'generate_behavioral_insights'
                ]
            }
        }
        
        # Export to JSON file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return f"Analysis report exported to {output_path}"

# Generator function for streaming analysis results
def stream_analysis_results(patterns: List[ActivityPattern]):
    """
    Generator function to stream analysis results
    Demonstrates: generators, yield statements, memory efficiency
    """
    for i, pattern in enumerate(patterns):
        yield {
            'index': i,
            'pattern': pattern,
            'percentage_complete': round((i + 1) / len(patterns) * 100, 2)
        }

# Decorator for caching expensive computations
def cache_result(func):
    """
    Decorator to cache function results
    Demonstrates: decorators, caching patterns
    """
    cache = {}
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

# Context manager for database operations
class DatabaseConnection:
    """
    Context manager for database connections
    Demonstrates: context managers, resource management
    """
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
    
    def __enter__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string)
        return self.client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

# Advanced Python data processing pipeline
class DataProcessingPipeline:
    """
    Data processing pipeline using Python advanced features
    Demonstrates: pipeline pattern, method chaining, functional programming
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def filter_by_confidence(self, min_confidence: float = 0.7):
        """Filter data by confidence threshold"""
        self.data = self.data[self.data['confidence'] >= min_confidence]
        return self
    
    def filter_by_activity_type(self, activity_types: List[str]):
        """Filter data by specific activity types"""
        self.data = self.data[self.data['activity_type'].isin(activity_types)]
        return self
    
    def add_time_features(self):
        """Add time-based features"""
        self.data['is_weekend'] = self.data['timestamp'].dt.dayofweek >= 5
        self.data['time_period'] = pd.cut(self.data['hour'], 
                                        bins=[0, 6, 12, 18, 24], 
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        return self
    
    def aggregate_by_date(self):
        """Aggregate activities by date"""
        self.data = self.data.groupby('date').agg({
            'activity_type': 'count',
            'confidence': 'mean',
            'behavioral_category': lambda x: x.value_counts().index[0]
        }).reset_index()
        return self
    
    def get_result(self) -> pd.DataFrame:
        """Get final processed data"""
        return self.data

# Example usage function
async def main_analysis_demo():
    """
    Demonstration of the Python data analysis capabilities
    """
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    
    # Using async context manager
    async with AutismDataAnalyzer(mongo_url) as analyzer:
        print("🐍 Starting Python Data Analysis Demo...")
        
        # Fetch and analyze data
        df = await analyzer.fetch_activities_data(days_back=7)
        print(f"📊 Fetched {len(df)} activity records")
        
        if not df.empty:
            # Generate patterns
            patterns = analyzer.generate_activity_patterns(df)
            print(f"🔍 Identified {len(patterns)} activity patterns")
            
            # Stream results using generator
            print("📈 Streaming analysis results:")
            for result in stream_analysis_results(patterns[:3]):
                print(f"  Progress: {result['percentage_complete']}% - {result['pattern'].activity_type}")
            
            # Generate insights
            insights = analyzer.generate_behavioral_insights(patterns)
            print(f"🧠 Generated {len(insights['insights'])} behavioral insights")
            
            # Export comprehensive report
            report_path = await analyzer.export_analysis_report()
            print(f"📋 {report_path}")
            
        else:
            print("⚠️ No data available for analysis")

if __name__ == "__main__":
    # Run the analysis demo
    asyncio.run(main_analysis_demo())