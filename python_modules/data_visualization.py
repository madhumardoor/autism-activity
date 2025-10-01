"""
Advanced Data Visualization Module for Autism Activity Recognition
Demonstrates: matplotlib, seaborn, plotly, interactive dashboards, statistical plots
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import motor.motor_asyncio
import os
from pathlib import Path
import base64
from io import BytesIO
import json
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ActivityVisualizationEngine:
    """
    Comprehensive visualization engine for autism activity data
    Demonstrates: Advanced plotting, statistical visualization, interactive charts
    """
    
    def __init__(self, db_connection_string: str, output_dir: str = "/app/visualizations"):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(db_connection_string)
        self.db = self.client[os.environ.get('DB_NAME', 'autism_monitoring_db')]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different categories
        self.behavior_colors = {
            'Calm': '#2E8B57',      # Sea Green
            'Active': '#FF6B35',     # Orange Red
            'Stimming': '#4169E1',   # Royal Blue
            'Concerning': '#DC143C', # Crimson
            'Therapeutic': '#9370DB' # Medium Purple
        }
        
        self.activity_colors = {
            'sitting': '#90EE90',
            'standing': '#87CEEB', 
            'walking': '#DDA0DD',
            'running': '#F0E68C',
            'jumping': '#FFB6C1',
            'hand_flapping': '#87CEFA',
            'rocking': '#98FB98',
            'spinning': '#F5DEB3',
            'head_banging': '#FFB6C1',
            'wandering': '#CD853F',
            'aggressive_behavior': '#DC143C',
            'self_harm': '#8B0000',
            'focused_activity': '#9370DB',
            'social_interaction': '#20B2AA',
            'therapy_exercise': '#32CD32'
        }
    
    async def fetch_visualization_data(self, days_back: int = 30) -> pd.DataFrame:
        """Fetch and prepare data for visualization"""
        start_date = datetime.now() - timedelta(days=days_back)
        
        cursor = self.db.activities.find({
            "timestamp": {"$gte": start_date}
        })
        
        activities = await cursor.to_list(length=None)
        
        if not activities:
            return pd.DataFrame()
        
        # Convert to DataFrame with comprehensive feature engineering
        df = pd.DataFrame(activities)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        
        # Behavioral categorization
        behavioral_map = {
            'sitting': 'Calm', 'standing': 'Calm',
            'walking': 'Active', 'running': 'Active', 'jumping': 'Active',
            'hand_flapping': 'Stimming', 'rocking': 'Stimming', 'spinning': 'Stimming',
            'head_banging': 'Concerning', 'wandering': 'Concerning', 
            'aggressive_behavior': 'Concerning', 'self_harm': 'Concerning',
            'focused_activity': 'Therapeutic', 'social_interaction': 'Therapeutic', 
            'therapy_exercise': 'Therapeutic'
        }
        
        df['behavioral_category'] = df['activity_type'].map(behavioral_map)
        
        # Time-based features
        df['time_period'] = pd.cut(df['hour'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        
        return df
    
    def create_activity_timeline(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create comprehensive activity timeline visualization
        Demonstrates: Time series plotting, multi-panel layouts, annotation
        """
        if df.empty:
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Autism Activity Recognition - Timeline Analysis', fontsize=16, fontweight='bold')
        
        # 1. Activity frequency over time
        daily_counts = df.groupby('date').size()
        axes[0].plot(daily_counts.index, daily_counts.values, 
                    marker='o', linewidth=2, markersize=6, color='#2E8B57')
        axes[0].fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='#2E8B57')
        axes[0].set_title('Daily Activity Frequency', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Activities')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = np.arange(len(daily_counts))
        z = np.polyfit(x_numeric, daily_counts.values, 1)
        p = np.poly1d(z)
        axes[0].plot(daily_counts.index, p(x_numeric), 
                    linestyle='--', color='red', alpha=0.8, label=f'Trend')
        axes[0].legend()
        
        # 2. Behavioral category distribution over time
        behavioral_pivot = df.groupby(['date', 'behavioral_category']).size().unstack(fill_value=0)
        
        bottom = np.zeros(len(behavioral_pivot))
        for category in behavioral_pivot.columns:
            axes[1].bar(behavioral_pivot.index, behavioral_pivot[category], 
                       bottom=bottom, label=category, 
                       color=self.behavior_colors.get(category, '#808080'), alpha=0.8)
            bottom += behavioral_pivot[category]
        
        axes[1].set_title('Behavioral Category Distribution Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Activities')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Confidence scores over time
        daily_confidence = df.groupby('date')['confidence'].agg(['mean', 'std']).fillna(0)
        
        axes[2].errorbar(daily_confidence.index, daily_confidence['mean'], 
                        yerr=daily_confidence['std'], 
                        marker='s', linewidth=2, markersize=4, 
                        capsize=5, color='#4169E1', alpha=0.8)
        axes[2].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Confidence (0.8+)')
        axes[2].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence (0.6+)')
        
        axes[2].set_title('Activity Detection Confidence Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Confidence Score')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            if len(daily_counts) > 10:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'activity_timeline.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def create_heatmap_analysis(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create heatmap visualizations for activity patterns
        Demonstrates: Heatmaps, correlation analysis, pattern recognition
        """
        if df.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Activity Pattern Heatmaps', fontsize=16, fontweight='bold')
        
        # 1. Hour vs Day of Week heatmap
        hour_dow_pivot = df.groupby(['hour', 'day_of_week']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hour_dow_pivot = hour_dow_pivot.reindex(columns=day_order, fill_value=0)
        
        sns.heatmap(hour_dow_pivot, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=axes[0,0], cbar_kws={'label': 'Activity Count'})
        axes[0,0].set_title('Activity Frequency by Hour and Day of Week')
        axes[0,0].set_ylabel('Hour of Day')
        
        # 2. Activity type correlation heatmap
        activity_hour_pivot = df.groupby(['activity_type', 'hour']).size().unstack(fill_value=0)
        
        # Calculate correlation matrix
        correlation_matrix = activity_hour_pivot.T.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=axes[0,1],
                   cbar_kws={'label': 'Correlation'})
        axes[0,1].set_title('Activity Type Correlation Matrix')
        
        # 3. Behavioral category by time period
        behavior_time_pivot = df.groupby(['behavioral_category', 'time_period']).size().unstack(fill_value=0)
        
        sns.heatmap(behavior_time_pivot, annot=True, fmt='d', cmap='viridis', 
                   ax=axes[1,0], cbar_kws={'label': 'Activity Count'})
        axes[1,0].set_title('Behavioral Categories by Time Period')
        
        # 4. Confidence by activity type and behavioral category
        confidence_pivot = df.groupby(['activity_type', 'behavioral_category'])['confidence'].mean().unstack()
        
        sns.heatmap(confidence_pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                   ax=axes[1,1], cbar_kws={'label': 'Avg Confidence'})
        axes[1,1].set_title('Average Confidence by Activity Type and Behavior')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'heatmap_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def create_statistical_analysis(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create statistical analysis visualizations
        Demonstrates: Statistical plots, distribution analysis, box plots, violin plots
        """
        if df.empty:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis of Activity Data', fontsize=16, fontweight='bold')
        
        # 1. Confidence distribution by behavioral category
        sns.violinplot(data=df, x='behavioral_category', y='confidence', 
                      palette=self.behavior_colors, ax=axes[0,0])
        axes[0,0].set_title('Confidence Distribution by Behavioral Category')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Activity frequency distribution
        activity_counts = df['activity_type'].value_counts()
        axes[0,1].bar(range(len(activity_counts)), activity_counts.values, 
                     color=[self.activity_colors.get(act, '#808080') for act in activity_counts.index])
        axes[0,1].set_title('Activity Type Frequency Distribution')
        axes[0,1].set_xticks(range(len(activity_counts)))
        axes[0,1].set_xticklabels(activity_counts.index, rotation=45, ha='right')
        axes[0,1].set_ylabel('Count')
        
        # Add frequency labels on bars
        for i, v in enumerate(activity_counts.values):
            axes[0,1].text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        # 3. Hourly activity distribution
        hourly_dist = df.groupby('hour').size()
        axes[0,2].plot(hourly_dist.index, hourly_dist.values, marker='o', linewidth=3, markersize=8)
        axes[0,2].fill_between(hourly_dist.index, hourly_dist.values, alpha=0.3)
        axes[0,2].set_title('Hourly Activity Distribution')
        axes[0,2].set_xlabel('Hour of Day')
        axes[0,2].set_ylabel('Activity Count')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Confidence vs Time scatter plot
        df_sample = df.sample(min(1000, len(df)))  # Sample for performance
        scatter = axes[1,0].scatter(df_sample['hour'], df_sample['confidence'], 
                                  c=[self.behavior_colors.get(cat, '#808080') 
                                     for cat in df_sample['behavioral_category']], 
                                  alpha=0.6, s=50)
        axes[1,0].set_title('Confidence vs Hour of Day')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Confidence Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['hour'], df['confidence'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(df['hour'].unique(), p(df['hour'].unique()), 
                      "r--", alpha=0.8, linewidth=2)
        
        # 5. Box plot of confidence by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sns.boxplot(data=df, x='day_of_week', y='confidence', 
                   order=day_order, ax=axes[1,1])
        axes[1,1].set_title('Confidence Distribution by Day of Week')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Activity duration analysis (simulated)
        # Calculate time differences between consecutive activities
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds() / 60  # minutes
        time_diffs = time_diffs.dropna()
        time_diffs = time_diffs[time_diffs < 120]  # Remove outliers > 2 hours
        
        if len(time_diffs) > 0:
            axes[1,2].hist(time_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,2].axvline(time_diffs.mean(), color='red', linestyle='--', 
                            label=f'Mean: {time_diffs.mean():.1f} min')
            axes[1,2].axvline(time_diffs.median(), color='green', linestyle='--', 
                            label=f'Median: {time_diffs.median():.1f} min')
            axes[1,2].set_title('Activity Duration Distribution')
            axes[1,2].set_xlabel('Minutes Between Activities')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'statistical_analysis.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create interactive Plotly dashboard
        Demonstrates: Interactive plotting, Plotly subplots, web-based visualization
        """
        if df.empty:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Activity Timeline', 'Behavioral Categories', 
                          'Confidence Distribution', 'Activity Heatmap',
                          'Weekly Patterns', 'Activity Network'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "heatmap"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.08
        )
        
        # 1. Activity timeline with confidence
        daily_data = df.groupby('date').agg({
            'activity_type': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=daily_data['date'], y=daily_data['activity_type'],
                      mode='lines+markers', name='Activity Count',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily_data['date'], y=daily_data['confidence'],
                      mode='lines+markers', name='Avg Confidence',
                      line=dict(color='red', width=2), yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Behavioral categories pie chart
        behavior_counts = df['behavioral_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=behavior_counts.index, values=behavior_counts.values,
                  marker_colors=[self.behavior_colors.get(cat, '#808080') 
                               for cat in behavior_counts.index]),
            row=1, col=2
        )
        
        # 3. Confidence distribution box plot
        for category in df['behavioral_category'].unique():
            category_data = df[df['behavioral_category'] == category]
            fig.add_trace(
                go.Box(y=category_data['confidence'], name=category,
                      marker_color=self.behavior_colors.get(category, '#808080')),
                row=2, col=1
            )
        
        # 4. Activity heatmap
        hour_activity_pivot = df.groupby(['hour', 'activity_type']).size().unstack(fill_value=0)
        
        fig.add_trace(
            go.Heatmap(z=hour_activity_pivot.values,
                      x=hour_activity_pivot.columns,
                      y=hour_activity_pivot.index,
                      colorscale='Viridis'),
            row=2, col=2
        )
        
        # 5. Weekly pattern analysis
        weekly_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_data = weekly_data.reindex(day_order, fill_value=0)
        
        fig.add_trace(
            go.Heatmap(z=weekly_data.values,
                      x=list(range(24)),
                      y=day_order,
                      colorscale='RdYlBu_r'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Autism Activity Dashboard",
            title_font_size=20,
            height=1200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Activity Count", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", secondary_y=True, row=1, col=1)
        
        fig.update_xaxes(title_text="Activity Type", row=2, col=2)
        fig.update_yaxes(title_text="Hour of Day", row=2, col=2)
        
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
        fig.update_yaxes(title_text="Day of Week", row=3, col=1)
        
        if save_path is None:
            save_path = self.output_dir / 'interactive_dashboard.html'
        
        fig.write_html(save_path)
        
        return str(save_path)
    
    def create_advanced_analytics(self, df: pd.DataFrame, save_path: str = None) -> str:
        """
        Create advanced analytics visualizations
        Demonstrates: PCA, t-SNE, clustering visualization, dimensionality reduction
        """
        if df.empty or len(df) < 10:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Analytics and Machine Learning Insights', fontsize=16, fontweight='bold')
        
        # Prepare features for ML analysis
        features_df = df.copy()
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        le_activity = LabelEncoder()
        le_behavior = LabelEncoder()
        
        features_df['activity_encoded'] = le_activity.fit_transform(df['activity_type'])
        features_df['behavior_encoded'] = le_behavior.fit_transform(df['behavioral_category'])
        
        # Select numerical features
        feature_columns = ['confidence', 'hour', 'activity_encoded', 'behavior_encoded']
        X = features_df[feature_columns].values
        
        # 1. PCA Analysis
        if len(df) >= 4:  # Need at least as many samples as features
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Color by behavioral category
            colors = [self.behavior_colors.get(cat, '#808080') for cat in df['behavioral_category']]
            scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=50)
            axes[0,0].set_title(f'PCA Analysis\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
            axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
            axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Activity clustering using K-means
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (up to 5)
        max_clusters = min(5, len(np.unique(df['behavioral_category'])))
        if len(df) >= max_clusters:
            kmeans = KMeans(n_clusters=max_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Plot clusters in PCA space
            if len(df) >= 4:
                X_pca_scaled = pca.transform(X_scaled)
                scatter = axes[0,1].scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], 
                                          c=clusters, cmap='tab10', alpha=0.7, s=50)
                axes[0,1].scatter(pca.transform(scaler.transform(kmeans.cluster_centers_))[:, 0],
                                pca.transform(scaler.transform(kmeans.cluster_centers_))[:, 1],
                                c='red', marker='x', s=200, linewidths=3)
                axes[0,1].set_title('K-Means Clustering in PCA Space')
                axes[0,1].set_xlabel('PC1')
                axes[0,1].set_ylabel('PC2')
                axes[0,1].grid(True, alpha=0.3)
        
        # 3. Feature importance analysis
        feature_importance = np.abs(pca.components_).mean(axis=0) if len(df) >= 4 else np.ones(len(feature_columns))
        feature_names = ['Confidence', 'Hour', 'Activity Type', 'Behavior Category']
        
        bars = axes[1,0].bar(feature_names, feature_importance, 
                           color=['#FF6B35', '#4169E1', '#32CD32', '#DC143C'])
        axes[1,0].set_title('Feature Importance (PCA Components)')
        axes[1,0].set_ylabel('Importance Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{importance:.3f}', ha='center', va='bottom')
        
        # 4. Confidence prediction analysis
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        if len(df) >= 10:
            # Prepare data for confidence prediction
            X_conf = features_df[['hour', 'activity_encoded', 'behavior_encoded']].values
            y_conf = features_df['confidence'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_conf, y_conf, test_size=0.3, random_state=42)
            
            # Train model
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            
            # Plot actual vs predicted
            axes[1,1].scatter(y_test, y_pred, alpha=0.7, color='blue', s=50)
            axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                          'r--', linewidth=2, alpha=0.8)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            axes[1,1].set_title(f'Confidence Prediction\nR² = {r2:.3f}, MSE = {mse:.3f}')
            axes[1,1].set_xlabel('Actual Confidence')
            axes[1,1].set_ylabel('Predicted Confidence')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'advanced_analytics.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    async def generate_comprehensive_report(self, days_back: int = 30) -> Dict[str, str]:
        """
        Generate comprehensive visualization report
        Demonstrates: Complete visualization pipeline, report generation
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Generating comprehensive visualization report for last {days_back} days...")
        
        # Fetch data
        df = await self.fetch_visualization_data(days_back)
        
        if df.empty:
            logger.warning("No data available for visualization")
            return {"error": "No data available"}
        
        logger.info(f"Processing {len(df)} activity records...")
        
        # Generate all visualizations
        report_paths = {}
        
        try:
            # 1. Timeline Analysis
            timeline_path = self.create_activity_timeline(df)
            if timeline_path:
                report_paths['timeline'] = timeline_path
            
            # 2. Heatmap Analysis
            heatmap_path = self.create_heatmap_analysis(df)
            if heatmap_path:
                report_paths['heatmaps'] = heatmap_path
            
            # 3. Statistical Analysis
            stats_path = self.create_statistical_analysis(df)
            if stats_path:
                report_paths['statistics'] = stats_path
            
            # 4. Interactive Dashboard
            dashboard_path = self.create_interactive_dashboard(df)
            if dashboard_path:
                report_paths['dashboard'] = dashboard_path
            
            # 5. Advanced Analytics
            advanced_path = self.create_advanced_analytics(df)
            if advanced_path:
                report_paths['advanced'] = advanced_path
            
            # Generate summary statistics
            summary_stats = {
                'total_activities': len(df),
                'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
                'unique_activity_types': df['activity_type'].nunique(),
                'avg_confidence': df['confidence'].mean(),
                'most_common_activity': df['activity_type'].mode().iloc[0] if not df['activity_type'].mode().empty else 'None',
                'behavioral_categories': df['behavioral_category'].nunique(),
                'visualizations_generated': len(report_paths)
            }
            
            # Save summary to JSON
            summary_path = self.output_dir / 'visualization_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            report_paths['summary'] = str(summary_path)
            
            logger.info(f"Visualization report generated successfully: {len(report_paths)} files created")
            
        except Exception as e:
            logger.error(f"Error generating visualization report: {str(e)}")
            report_paths['error'] = str(e)
        
        finally:
            # Close database connection
            self.client.close()
        
        return report_paths

# Example usage function
async def visualization_demo():
    """
    Demonstration of the visualization capabilities
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize visualization engine
    db_connection = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    viz_engine = ActivityVisualizationEngine(db_connection)
    
    print("🎨 Starting Visualization Demo...")
    
    # Generate comprehensive report
    report_paths = await viz_engine.generate_comprehensive_report(days_back=7)
    
    if 'error' not in report_paths:
        print("📊 Visualization Report Generated:")
        for viz_type, path in report_paths.items():
            print(f"  {viz_type.title()}: {path}")
    else:
        print(f"❌ Error: {report_paths['error']}")

if __name__ == "__main__":
    # Ensure output directory exists
    Path("/app/visualizations").mkdir(parents=True, exist_ok=True)
    
    # Run the visualization demo
    asyncio.run(visualization_demo())