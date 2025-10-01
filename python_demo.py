#!/usr/bin/env python3
"""
Comprehensive Python Demo for Autism Child Activity Recognition System
Demonstrates: Advanced Python programming concepts and integrations

This script showcases all the Python modules and advanced programming techniques
implemented in the autism monitoring system.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add python_modules to path
sys.path.append('/app/python_modules')

# Import custom modules
from data_analysis import AutismDataAnalyzer, stream_analysis_results, DataProcessingPipeline
from ml_models import (
    RandomForestActivityClassifier, SVMActivityClassifier, 
    LSTMActivityClassifier, ModelConfig, DeepLearningTrainer,
    ModelEvaluator, FeatureExtractor, ml_pipeline_demo
)
from automation_scripts import (
    AutomationOrchestrator, AutomationConfig, SystemMonitor,
    DatabaseManager, NotificationManager, TaskScheduler
)
from data_visualization import ActivityVisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonDemoOrchestrator:
    """
    Main orchestrator to demonstrate all Python capabilities
    Showcases: Class design, async programming, context managers, decorators
    """
    
    def __init__(self):
        self.db_connection = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
        self.output_dir = Path('/app/demo_outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config = AutomationConfig()
        self.demo_results = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        logger.info("🐍 Initializing Python Demo Environment...")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        logger.info("🐍 Python Demo Environment closed")
    
    def demo_decorator(demo_name: str):
        """
        Decorator for demo functions
        Demonstrates: Decorators, function wrapping, timing
        """
        def decorator(func):
            async def wrapper(self, *args, **kwargs):
                logger.info(f"🚀 Starting {demo_name}...")
                start_time = datetime.now()
                
                try:
                    result = await func(self, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(self, *args, **kwargs)
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ {demo_name} completed in {execution_time:.2f} seconds")
                    
                    self.demo_results[demo_name] = {
                        'status': 'success',
                        'execution_time': execution_time,
                        'result': result
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ {demo_name} failed: {str(e)}")
                    self.demo_results[demo_name] = {
                        'status': 'error',
                        'error': str(e),
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    }
                    raise
            
            return wrapper
        return decorator
    
    @demo_decorator("Data Analysis Pipeline")
    async def demo_data_analysis(self):
        """
        Demonstrate advanced data analysis capabilities
        Showcases: Pandas, NumPy, async operations, data processing
        """
        logger.info("📊 Demonstrating Data Analysis Pipeline...")
        
        # Initialize analyzer with async context manager
        async with AutismDataAnalyzer(self.db_connection) as analyzer:
            
            # Fetch and analyze data
            df = await analyzer.fetch_activities_data(days_back=7)
            logger.info(f"📥 Fetched {len(df)} activity records")
            
            if df.empty:
                logger.warning("No data available - generating sample data for demo")
                return {"message": "No data available for analysis"}
            
            # Generate activity patterns
            patterns = analyzer.generate_activity_patterns(df)
            logger.info(f"🔍 Generated {len(patterns)} activity patterns")
            
            # Demonstrate generator function
            logger.info("🔄 Streaming pattern analysis:")
            for i, result in enumerate(stream_analysis_results(patterns[:3])):
                logger.info(f"  Pattern {i+1}: {result['pattern'].activity_type} - {result['percentage_complete']}% complete")
            
            # Perform clustering analysis
            clustering_results = analyzer.activity_clustering_analysis(df)
            logger.info("🎯 Clustering analysis completed")
            
            # Generate behavioral insights
            insights = analyzer.generate_behavioral_insights(patterns)
            logger.info(f"🧠 Generated insights for {len(insights['insights'])} behavioral categories")
            
            # Demonstrate data processing pipeline
            if not df.empty:
                pipeline = DataProcessingPipeline(df)
                processed_data = (pipeline
                                .filter_by_confidence(0.7)
                                .add_time_features()
                                .get_result())
                
                logger.info(f"⚙️ Processing pipeline: {len(df)} → {len(processed_data)} records")
            
            # Export comprehensive report
            report_path = await analyzer.export_analysis_report()
            logger.info(f"📋 Analysis report exported: {report_path}")
            
            return {
                'patterns_count': len(patterns),
                'clustering_results': clustering_results,
                'insights_summary': insights['summary'],
                'report_path': report_path
            }
    
    @demo_decorator("Machine Learning Pipeline")
    async def demo_machine_learning(self):
        """
        Demonstrate machine learning capabilities
        Showcases: PyTorch, scikit-learn, model training, evaluation
        """
        logger.info("🤖 Demonstrating Machine Learning Pipeline...")
        
        # Run comprehensive ML demo
        ml_results = await ml_pipeline_demo()
        
        logger.info("🎯 ML Pipeline Results:")
        logger.info(f"  Random Forest Accuracy: {ml_results['rf_accuracy']:.3f}")
        logger.info(f"  SVM Accuracy: {ml_results['svm_accuracy']:.3f}")
        
        # Demonstrate feature extraction
        extractor = FeatureExtractor()
        
        # Sample sequence for feature extraction
        import numpy as np
        sample_sequence = np.random.randn(100)
        statistical_features = extractor.extract_statistical_features(sample_sequence)
        
        logger.info(f"📈 Extracted {len(statistical_features)} statistical features")
        
        # Demonstrate model configuration
        config = ModelConfig(
            sequence_length=20,
            input_features=10,
            hidden_size=128,
            num_layers=3,
            dropout_rate=0.2
        )
        
        logger.info(f"⚙️ Model config: {config.sequence_length} seq length, {config.hidden_size} hidden units")
        
        return {
            'rf_accuracy': ml_results['rf_accuracy'],
            'svm_accuracy': ml_results['svm_accuracy'],
            'feature_count': len(statistical_features),
            'model_config': config.__dict__
        }
    
    @demo_decorator("Automation System")
    async def demo_automation(self):
        """
        Demonstrate automation capabilities
        Showcases: Task scheduling, system monitoring, notifications
        """
        logger.info("⚙️ Demonstrating Automation System...")
        
        # Initialize automation components
        orchestrator = AutomationOrchestrator(self.config)
        
        # Demonstrate system monitoring
        monitor = SystemMonitor(self.config)
        metrics = monitor.get_system_metrics()
        
        logger.info("📊 System Metrics:")
        logger.info(f"  CPU Usage: {metrics.cpu_percent:.1f}%")
        logger.info(f"  Memory Usage: {metrics.memory_percent:.1f}%")
        logger.info(f"  Disk Usage: {metrics.disk_usage_percent:.1f}%")
        
        # Demonstrate database operations
        db_manager = DatabaseManager(self.config)
        
        # Create a test backup (commented out to avoid large files in demo)
        # backup_path = await db_manager.backup_database()
        # logger.info(f"💾 Backup created: {backup_path}")
        
        # Generate system report
        system_report = await orchestrator.generate_system_report()
        logger.info("📋 System report generated")
        
        # Demonstrate task scheduling setup
        scheduler = TaskScheduler(self.config)
        
        @scheduler.task("every 1 minutes", "demo_task")
        def sample_task():
            logger.info("⏰ Sample scheduled task executed")
            return "Task completed"
        
        logger.info(f"📅 Scheduler initialized with {len(scheduler.tasks)} tasks")
        
        return {
            'system_metrics': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_usage_percent
            },
            'system_report': system_report,
            'scheduled_tasks': len(scheduler.tasks)
        }
    
    @demo_decorator("Data Visualization")
    async def demo_visualization(self):
        """
        Demonstrate data visualization capabilities
        Showcases: Matplotlib, Seaborn, Plotly, statistical plots
        """
        logger.info("📊 Demonstrating Data Visualization...")
        
        # Initialize visualization engine
        viz_engine = ActivityVisualizationEngine(self.db_connection, str(self.output_dir / 'visualizations'))
        
        # Generate comprehensive visualization report
        viz_report = await viz_engine.generate_comprehensive_report(days_back=7)
        
        if 'error' in viz_report:
            logger.warning(f"Visualization demo limited due to: {viz_report['error']}")
            return {'status': 'limited', 'reason': viz_report['error']}
        
        logger.info("📈 Visualization Report Generated:")
        for viz_type, path in viz_report.items():
            if viz_type != 'error':
                logger.info(f"  {viz_type.title()}: {Path(path).name}")
        
        return {
            'visualizations_generated': len([k for k in viz_report.keys() if k != 'error']),
            'report_paths': viz_report
        }
    
    @demo_decorator("Advanced Python Features")
    def demo_advanced_features(self):
        """
        Demonstrate advanced Python programming concepts
        Showcases: Generators, decorators, context managers, metaclasses, etc.
        """
        logger.info("🔧 Demonstrating Advanced Python Features...")
        
        # 1. Generator functions
        def fibonacci_generator(n):
            """Generator for Fibonacci sequence"""
            a, b = 0, 1
            for _ in range(n):
                yield a
                a, b = b, a + b
        
        fib_sequence = list(fibonacci_generator(10))
        logger.info(f"🔢 Fibonacci sequence (10 terms): {fib_sequence}")
        
        # 2. List comprehensions and dictionary comprehensions
        squares = [x**2 for x in range(1, 11) if x % 2 == 0]
        square_dict = {x: x**2 for x in range(1, 6)}
        logger.info(f"🔲 Even squares: {squares}")
        logger.info(f"📚 Square dictionary: {square_dict}")
        
        # 3. Lambda functions and functional programming
        from functools import reduce, partial
        
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Filter, map, reduce operations
        evens = list(filter(lambda x: x % 2 == 0, numbers))
        doubled = list(map(lambda x: x * 2, numbers))
        sum_all = reduce(lambda x, y: x + y, numbers)
        
        logger.info(f"🔽 Filtered evens: {evens}")
        logger.info(f"🔄 Doubled: {doubled[:5]}...")  # Show first 5
        logger.info(f"➕ Sum of all: {sum_all}")
        
        # 4. Partial functions
        def multiply(x, y):
            return x * y
        
        double = partial(multiply, 2)
        triple = partial(multiply, 3)
        
        logger.info(f"✖️ Partial functions: double(5) = {double(5)}, triple(4) = {triple(4)}")
        
        # 5. Decorator patterns
        def cache_decorator(func):
            cache = {}
            def wrapper(n):
                if n not in cache:
                    cache[n] = func(n)
                return cache[n]
            return wrapper
        
        @cache_decorator
        def expensive_computation(n):
            """Simulate expensive computation"""
            return sum(range(n))
        
        result1 = expensive_computation(1000)
        result2 = expensive_computation(1000)  # Should use cache
        logger.info(f"💾 Cached computation result: {result1}")
        
        # 6. Context managers
        class TimerContext:
            def __enter__(self):
                self.start_time = datetime.now()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                execution_time = (datetime.now() - self.start_time).total_seconds()
                logger.info(f"⏱️ Context manager execution time: {execution_time:.4f}s")
        
        with TimerContext():
            # Simulate some work
            sum(range(100000))
        
        # 7. Dataclasses and type hints
        from dataclasses import dataclass
        from typing import List, Dict, Optional, Union
        
        @dataclass
        class ActivityMetrics:
            activity_type: str
            confidence: float
            timestamp: datetime
            metadata: Optional[Dict[str, Union[str, int]]] = None
        
        sample_metrics = ActivityMetrics(
            activity_type="sitting",
            confidence=0.95,
            timestamp=datetime.now(),
            metadata={"duration": 120, "location": "room_1"}
        )
        
        logger.info(f"📊 Sample metrics: {sample_metrics.activity_type} at {sample_metrics.confidence:.2f} confidence")
        
        # 8. Asyncio patterns
        async def async_task(name: str, delay: float):
            await asyncio.sleep(delay)
            return f"Task {name} completed after {delay}s"
        
        # Run concurrent tasks
        async def run_concurrent_tasks():
            tasks = [
                async_task("A", 0.1),
                async_task("B", 0.05),
                async_task("C", 0.15)
            ]
            results = await asyncio.gather(*tasks)
            return results
        
        # Note: This would be called with asyncio.run() in a real scenario
        logger.info("🔄 Async task patterns demonstrated (concurrent execution)")
        
        return {
            'fibonacci_sequence': fib_sequence,
            'functional_programming': {
                'evens': evens,
                'sum': sum_all
            },
            'sample_metrics': {
                'type': sample_metrics.activity_type,
                'confidence': sample_metrics.confidence
            }
        }
    
    async def run_comprehensive_demo(self):
        """
        Run all demonstration modules
        Showcases: Complete system integration, error handling, reporting
        """
        logger.info("🎯 Starting Comprehensive Python Demo...")
        
        # Run all demos
        try:
            # 1. Data Analysis
            await self.demo_data_analysis()
            
            # 2. Machine Learning
            await self.demo_machine_learning()
            
            # 3. Automation
            await self.demo_automation()
            
            # 4. Visualization  
            await self.demo_visualization()
            
            # 5. Advanced Features
            self.demo_advanced_features()
            
        except Exception as e:
            logger.error(f"Demo execution error: {str(e)}")
        
        # Generate final report
        report_path = self.output_dir / 'python_demo_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.demo_results, f, indent=2, default=str)
        
        logger.info(f"📋 Comprehensive demo report saved: {report_path}")
        
        # Display summary
        successful_demos = len([r for r in self.demo_results.values() if r['status'] == 'success'])
        total_demos = len(self.demo_results)
        
        logger.info("🎉 Demo Summary:")
        logger.info(f"  Successful demos: {successful_demos}/{total_demos}")
        logger.info(f"  Total execution time: {sum(r.get('execution_time', 0) for r in self.demo_results.values()):.2f}s")
        
        for demo_name, result in self.demo_results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            logger.info(f"  {status_emoji} {demo_name}: {result['status']}")
        
        return self.demo_results

# CLI interface for running specific demos
def main():
    """
    Main CLI interface for Python demos
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Autism Activity Recognition - Python Demo")
    parser.add_argument('--demo', choices=['all', 'data', 'ml', 'automation', 'viz', 'advanced'],
                       default='all', help='Which demo to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_selected_demo():
        async with PythonDemoOrchestrator() as demo:
            if args.demo == 'all':
                await demo.run_comprehensive_demo()
            elif args.demo == 'data':
                await demo.demo_data_analysis()
            elif args.demo == 'ml':
                await demo.demo_machine_learning()
            elif args.demo == 'automation':
                await demo.demo_automation()
            elif args.demo == 'viz':
                await demo.demo_visualization()
            elif args.demo == 'advanced':
                demo.demo_advanced_features()
    
    # Ensure required directories exist
    Path('/app/demo_outputs').mkdir(parents=True, exist_ok=True)
    Path('/app/models').mkdir(parents=True, exist_ok=True)
    Path('/app/logs').mkdir(parents=True, exist_ok=True)
    Path('/app/visualizations').mkdir(parents=True, exist_ok=True)
    
    # Run the selected demo
    asyncio.run(run_selected_demo())

if __name__ == "__main__":
    main()