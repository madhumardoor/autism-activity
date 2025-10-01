"""
Python Automation Scripts for Autism Activity Recognition System
Demonstrates: Task automation, scheduling, monitoring, data processing
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from pathlib import Path
import shutil
import psutil
import motor.motor_asyncio
from contextlib import asynccontextmanager
import subprocess
import yaml
from collections import defaultdict, deque
from functools import wraps, partial
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AutomationConfig:
    """Configuration class for automation settings"""
    db_connection: str = "mongodb://localhost:27017"
    db_name: str = "autism_monitoring_db"
    backup_directory: str = "/app/backups"
    log_directory: str = "/app/logs"
    alert_threshold: int = 5
    monitoring_interval: int = 300  # seconds
    cleanup_days: int = 30
    email_notifications: bool = False
    system_monitoring: bool = True
    auto_backup: bool = True

@dataclass
class SystemMetrics:
    """System performance metrics data class"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    load_average: List[float] = field(default_factory=list)

class TaskScheduler:
    """
    Advanced task scheduler with cron-like functionality
    Demonstrates: Task scheduling, threading, async operations
    """
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.tasks = {}
        self.running = False
        self.scheduler_thread = None
        
    def task(self, schedule_time: str, name: Optional[str] = None):
        """
        Decorator for scheduling tasks
        Demonstrates: Decorators, function wrapping, scheduling
        """
        def decorator(func: Callable):
            task_name = name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    logger.info(f"Executing scheduled task: {task_name}")
                    start_time = time.time()
                    
                    # Handle both sync and async functions
                    if inspect.iscoroutinefunction(func):
                        result = asyncio.run(func(*args, **kwargs))
                    else:
                        result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    logger.info(f"Task {task_name} completed in {execution_time:.2f}s")
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in scheduled task {task_name}: {str(e)}")
                    raise
            
            # Schedule the task
            if schedule_time == "hourly":
                schedule.every().hour.do(wrapper)
            elif schedule_time == "daily":
                schedule.every().day.do(wrapper)
            elif schedule_time.startswith("every"):
                # Parse "every X minutes/hours"
                parts = schedule_time.split()
                interval = int(parts[1])
                unit = parts[2]
                if "minute" in unit:
                    schedule.every(interval).minutes.do(wrapper)
                elif "hour" in unit:
                    schedule.every(interval).hours.do(wrapper)
            
            self.tasks[task_name] = {
                'function': wrapper,
                'schedule': schedule_time,
                'last_run': None,
                'run_count': 0
            }
            
            return wrapper
        return decorator
    
    def start(self):
        """Start the task scheduler in a separate thread"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Task scheduler started")
    
    def stop(self):
        """Stop the task scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Task scheduler stopped")
    
    def _run_scheduler(self):
        """Internal method to run scheduled tasks"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)

class SystemMonitor:
    """
    System performance and health monitoring
    Demonstrates: System monitoring, performance metrics, alerting
    """
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for system alerts"""
        self.alert_callbacks.append(callback)
    
    def get_system_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics
        Demonstrates: System introspection, performance monitoring
        """
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network metrics
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io,
            process_count=process_count,
            load_average=load_avg
        )
        
        self.metrics_history.append(metrics)
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check for system alerts and trigger callbacks"""
        alerts = []
        
        if metrics.cpu_percent > 80:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent
            })
        
        if metrics.memory_percent > 85:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f'High memory usage: {metrics.memory_percent:.1f}%',
                'value': metrics.memory_percent
            })
        
        if metrics.disk_usage_percent > 90:
            alerts.append({
                'type': 'high_disk',
                'severity': 'critical',
                'message': f'High disk usage: {metrics.disk_usage_percent:.1f}%',
                'value': metrics.disk_usage_percent
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {str(e)}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for recent metrics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            'period_hours': hours,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'max': max(m.cpu_percent for m in recent_metrics),
                'min': min(m.cpu_percent for m in recent_metrics)
            },
            'memory': {
                'avg': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                'max': max(m.memory_percent for m in recent_metrics),
                'min': min(m.memory_percent for m in recent_metrics)
            },
            'disk': {
                'current': recent_metrics[-1].disk_usage_percent
            },
            'process_count': {
                'avg': sum(m.process_count for m in recent_metrics) / len(recent_metrics),
                'current': recent_metrics[-1].process_count
            }
        }

class DatabaseManager:
    """
    Database management and maintenance automation
    Demonstrates: Database operations, backup management, cleanup
    """
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.client = None
        self.db = None
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Async context manager for database connections"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.config.db_connection)
            self.db = self.client[self.config.db_name]
            yield self.db
        finally:
            if self.client:
                self.client.close()
    
    async def backup_database(self) -> str:
        """
        Create database backup
        Demonstrates: Database backup, file operations, async operations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(self.config.backup_directory)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_file = backup_dir / f"autism_db_backup_{timestamp}.json"
        
        async with self.get_db_connection() as db:
            # Backup activities collection
            activities = await db.activities.find().to_list(length=None)
            
            # Backup labeling data
            labeling_data = await db.labeling_data.find().to_list(length=None)
            
            backup_data = {
                'timestamp': timestamp,
                'activities': activities,
                'labeling_data': labeling_data,
                'metadata': {
                    'activities_count': len(activities),
                    'labeling_count': len(labeling_data)
                }
            }
            
            # Convert ObjectId to string for JSON serialization
            def convert_objectid(obj):
                if hasattr(obj, '_id'):
                    obj['_id'] = str(obj['_id'])
                return obj
            
            backup_data['activities'] = [convert_objectid(act) for act in activities]
            backup_data['labeling_data'] = [convert_objectid(data) for data in labeling_data]
        
        # Write backup file
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"Database backup created: {backup_file}")
        return str(backup_file)
    
    async def cleanup_old_data(self, days_to_keep: int = None) -> Dict[str, int]:
        """
        Clean up old data from the database
        Demonstrates: Data cleanup, date operations, bulk operations
        """
        days_to_keep = days_to_keep or self.config.cleanup_days
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        async with self.get_db_connection() as db:
            # Count records to be deleted
            old_activities = await db.activities.count_documents({
                "timestamp": {"$lt": cutoff_date}
            })
            
            old_labeling = await db.labeling_data.count_documents({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Delete old records
            activities_result = await db.activities.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            labeling_result = await db.labeling_data.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
        
        cleanup_stats = {
            'activities_deleted': activities_result.deleted_count,
            'labeling_deleted': labeling_result.deleted_count,
            'cutoff_date': cutoff_date.isoformat()
        }
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats

class NotificationManager:
    """
    Notification and alerting system
    Demonstrates: Email notifications, webhook calls, alert management
    """
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.email_config = self._load_email_config()
    
    def _load_email_config(self) -> Dict[str, str]:
        """Load email configuration from environment or config file"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME', ''),
            'password': os.getenv('EMAIL_PASSWORD', ''),
            'from_address': os.getenv('FROM_EMAIL', ''),
            'to_addresses': os.getenv('TO_EMAILS', '').split(',')
        }
    
    async def send_alert_email(self, subject: str, message: str, alert_data: Dict[str, Any] = None):
        """
        Send alert email notification
        Demonstrates: Email sending, HTML formatting, error handling
        """
        if not self.config.email_notifications or not self.email_config['username']:
            logger.info("Email notifications disabled or not configured")
            return
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[Autism Monitor] {subject}"
            msg['From'] = self.email_config['from_address']
            msg['To'] = ', '.join(self.email_config['to_addresses'])
            
            # Create HTML content
            html_content = self._create_alert_html(message, alert_data)
            html_part = MIMEText(html_content, 'html')
            text_part = MIMEText(message, 'plain')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            logger.info(f"Alert email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {str(e)}")
    
    def _create_alert_html(self, message: str, alert_data: Dict[str, Any] = None) -> str:
        """Create HTML formatted alert email"""
        html = f"""
        <html>
        <body>
            <h2>🚨 Autism Monitoring System Alert</h2>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        if alert_data:
            html += "<h3>Alert Details:</h3><ul>"
            for key, value in alert_data.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
            <p>Please check the system dashboard for more details.</p>
            <p><em>This is an automated message from the Autism Child Activity Recognition System.</em></p>
        </body>
        </html>
        """
        
        return html
    
    async def send_webhook_notification(self, webhook_url: str, data: Dict[str, Any]):
        """
        Send webhook notification
        Demonstrates: HTTP requests, JSON payloads, error handling
        """
        try:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'system': 'autism_monitor',
                'data': data
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent successfully to {webhook_url}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {str(e)}")

class AutomationOrchestrator:
    """
    Main orchestrator for all automation tasks
    Demonstrates: Orchestration patterns, dependency injection, lifecycle management
    """
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.scheduler = TaskScheduler(config)
        self.monitor = SystemMonitor(config)
        self.db_manager = DatabaseManager(config)
        self.notification_manager = NotificationManager(config)
        
        # Setup alert callbacks
        self.monitor.add_alert_callback(self._handle_system_alert)
        
        # Setup scheduled tasks
        self._setup_scheduled_tasks()
    
    def _setup_scheduled_tasks(self):
        """Setup all scheduled automation tasks"""
        
        @self.scheduler.task("hourly", "system_health_check")
        async def system_health_check():
            """Hourly system health monitoring"""
            metrics = self.monitor.get_system_metrics()
            logger.info(f"System Health - CPU: {metrics.cpu_percent:.1f}%, "
                       f"Memory: {metrics.memory_percent:.1f}%, "
                       f"Disk: {metrics.disk_usage_percent:.1f}%")
            
            # Generate daily summary
            if datetime.now().hour == 0:  # Midnight
                summary = self.monitor.get_metrics_summary(24)
                await self.notification_manager.send_alert_email(
                    "Daily System Summary",
                    "Daily system performance summary attached",
                    summary
                )
        
        @self.scheduler.task("daily", "database_backup")
        async def database_backup():
            """Daily database backup"""
            try:
                backup_file = await self.db_manager.backup_database()
                logger.info(f"Daily backup completed: {backup_file}")
            except Exception as e:
                logger.error(f"Daily backup failed: {str(e)}")
                await self.notification_manager.send_alert_email(
                    "Backup Failed",
                    f"Daily database backup failed: {str(e)}"
                )
        
        @self.scheduler.task("every 7 days", "cleanup_old_data")
        async def cleanup_old_data():
            """Weekly data cleanup"""
            try:
                cleanup_stats = await self.db_manager.cleanup_old_data()
                logger.info(f"Weekly cleanup completed: {cleanup_stats}")
            except Exception as e:
                logger.error(f"Weekly cleanup failed: {str(e)}")
        
        @self.scheduler.task("every 30 minutes", "activity_analysis")
        async def activity_analysis():
            """Regular activity pattern analysis"""
            from data_analysis import AutismDataAnalyzer
            
            try:
                async with AutismDataAnalyzer(self.config.db_connection) as analyzer:
                    df = await analyzer.fetch_activities_data(days_back=1)
                    
                    if not df.empty:
                        patterns = analyzer.generate_activity_patterns(df)
                        insights = analyzer.generate_behavioral_insights(patterns)
                        
                        # Check for concerning patterns
                        concerning_count = insights['summary'].get('concerning_behaviors', 0)
                        if concerning_count > self.config.alert_threshold:
                            await self.notification_manager.send_alert_email(
                                "Concerning Activity Alert",
                                f"Detected {concerning_count} concerning behavioral patterns",
                                insights['summary']
                            )
                        
                        logger.info(f"Activity analysis completed - {len(patterns)} patterns analyzed")
                    
            except Exception as e:
                logger.error(f"Activity analysis failed: {str(e)}")
    
    async def _handle_system_alert(self, alert: Dict[str, Any]):
        """Handle system performance alerts"""
        if alert['severity'] == 'critical':
            await self.notification_manager.send_alert_email(
                f"Critical System Alert: {alert['type']}",
                alert['message'],
                alert
            )
        elif alert['severity'] == 'warning':
            logger.warning(f"System Alert: {alert['message']}")
    
    def start(self):
        """Start the automation orchestrator"""
        logger.info("🤖 Starting Autism Monitor Automation System...")
        
        # Create necessary directories
        Path(self.config.backup_directory).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_directory).mkdir(parents=True, exist_ok=True)
        
        # Start scheduler
        self.scheduler.start()
        
        # Start system monitoring
        if self.config.system_monitoring:
            logger.info("System monitoring enabled")
        
        logger.info("✅ Automation system started successfully")
    
    def stop(self):
        """Stop the automation orchestrator"""
        logger.info("Stopping automation system...")
        self.scheduler.stop()
        logger.info("Automation system stopped")
    
    async def run_manual_backup(self) -> str:
        """Manually trigger a database backup"""
        return await self.db_manager.backup_database()
    
    async def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        metrics_summary = self.monitor.get_metrics_summary(24)
        
        # Get database statistics
        async with self.db_manager.get_db_connection() as db:
            activity_count = await db.activities.count_documents({})
            labeling_count = await db.labeling_data.count_documents({})
        
        return {
            'generated_at': datetime.now().isoformat(),
            'system_metrics': metrics_summary,
            'database_stats': {
                'total_activities': activity_count,
                'total_labeling_data': labeling_count
            },
            'scheduled_tasks': {
                'total_tasks': len(self.scheduler.tasks),
                'task_names': list(self.scheduler.tasks.keys())
            },
            'automation_config': {
                'monitoring_enabled': self.config.system_monitoring,
                'auto_backup_enabled': self.config.auto_backup,
                'email_notifications': self.config.email_notifications
            }
        }

# Main automation entry point
async def main():
    """
    Main entry point for automation system
    Demonstrates: Application lifecycle, configuration management
    """
    
    # Load configuration
    config = AutomationConfig()
    
    # Initialize orchestrator
    orchestrator = AutomationOrchestrator(config)
    
    try:
        # Start automation system
        orchestrator.start()
        
        # Run initial system report
        report = await orchestrator.generate_system_report()
        logger.info("Initial system report generated")
        
        # Keep the automation running
        logger.info("Automation system running... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Received stop signal")
    finally:
        orchestrator.stop()
        logger.info("Automation system shutdown complete")

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs('/app/logs', exist_ok=True)
    
    # Run the automation system
    asyncio.run(main())