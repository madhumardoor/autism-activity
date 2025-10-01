"""
Advanced Machine Learning Models for Autism Activity Recognition
Demonstrates: TensorFlow/Keras, PyTorch, scikit-learn, custom model architectures
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    sequence_length: int = 10
    input_features: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    num_classes: int = 15
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

class ActivityDataset(Dataset):
    """
    PyTorch Dataset class for activity recognition data
    Demonstrates: PyTorch data handling, custom dataset creation
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, transform=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label

class LSTMActivityClassifier(nn.Module):
    """
    LSTM Neural Network for Activity Classification
    Demonstrates: PyTorch neural networks, LSTM architecture, custom model design
    """
    
    def __init__(self, config: ModelConfig):
        super(LSTMActivityClassifier, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout_rate
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc2 = nn.Linear(config.hidden_size // 2, config.num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        self.batch_norm = nn.BatchNorm1d(config.hidden_size // 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for classification
        last_output = attention_out[:, -1, :]
        
        # Fully connected layers with regularization
        out = self.dropout(self.relu(self.fc1(last_output)))
        out = self.batch_norm(out)
        out = self.fc2(out)
        
        return out

class BaseMLModel(ABC):
    """
    Abstract base class for ML models
    Demonstrates: Abstract classes, inheritance, polymorphism
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Abstract method for training the model"""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Abstract method for making predictions"""
        pass
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_name': self.model_name
            }, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

class RandomForestActivityClassifier(BaseMLModel):
    """
    Random Forest classifier for activity recognition
    Demonstrates: Inheritance, scikit-learn ensemble methods
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Random Forest model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        self.model.fit(X_train_scaled, y_train_encoded)
        self.is_trained = True
        logger.info("Random Forest model trained successfully")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions_encoded = self.model.predict(X_test_scaled)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return {i: importance for i, importance in enumerate(self.model.feature_importances_)}

class SVMActivityClassifier(BaseMLModel):
    """
    Support Vector Machine classifier for activity recognition
    Demonstrates: SVM implementation, kernel methods
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__("SVM")
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the SVM model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        self.model.fit(X_train_scaled, y_train_encoded)
        self.is_trained = True
        logger.info("SVM model trained successfully")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions_encoded = self.model.predict(X_test_scaled)
        return self.label_encoder.inverse_transform(predictions_encoded)

class DeepLearningTrainer:
    """
    PyTorch deep learning trainer class
    Demonstrates: Deep learning training loops, optimization, validation
    """
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Full training loop with validation
        Demonstrates: Training loops, early stopping, model checkpointing
        """
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        logger.info("Starting deep learning training...")
        
        for epoch in range(self.config.epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/app/models/best_lstm_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f'Epoch [{epoch}/{self.config.epochs}], '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

class ModelEvaluator:
    """
    Model evaluation and comparison class
    Demonstrates: Model evaluation metrics, visualization, comparison
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model: BaseMLModel, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        predictions = model.predict(X_test)
        
        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        evaluation_result = {
            'model_name': model.model_name,
            'accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg'],
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist()
        }
        
        self.evaluation_results[model.model_name] = evaluation_result
        return evaluation_result
    
    def plot_confusion_matrix(self, model_name: str, class_names: List[str] = None):
        """Plot confusion matrix for a specific model"""
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'/app/models/{model_name}_confusion_matrix.png')
        plt.close()
        
        logger.info(f"Confusion matrix saved for {model_name}")
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models"""
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision (Macro)': results['macro_avg']['precision'],
                'Recall (Macro)': results['macro_avg']['recall'],
                'F1-Score (Macro)': results['macro_avg']['f1-score'],
                'F1-Score (Weighted)': results['weighted_avg']['f1-score']
            })
        
        return pd.DataFrame(comparison_data)

class FeatureExtractor:
    """
    Feature extraction class for activity data
    Demonstrates: Feature engineering, signal processing, statistical features
    """
    
    @staticmethod
    def extract_statistical_features(sequence: np.ndarray) -> np.ndarray:
        """Extract statistical features from activity sequences"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(sequence),
            np.std(sequence),
            np.min(sequence),
            np.max(sequence),
            np.median(sequence)
        ])
        
        # Percentiles
        features.extend([
            np.percentile(sequence, 25),
            np.percentile(sequence, 75)
        ])
        
        # Advanced statistics
        features.extend([
            np.var(sequence),
            np.ptp(sequence),  # Peak-to-peak
            len(sequence)
        ])
        
        return np.array(features)
    
    @staticmethod
    def extract_temporal_features(timestamps: List[str]) -> np.ndarray:
        """Extract temporal features from timestamps"""
        timestamps_dt = pd.to_datetime(timestamps)
        
        features = []
        
        # Time-based features
        features.extend([
            timestamps_dt.hour.mean(),
            timestamps_dt.hour.std(),
            timestamps_dt.dayofweek.mean(),
            len(set(timestamps_dt.date))  # Number of unique days
        ])
        
        # Temporal patterns
        time_diffs = timestamps_dt.diff().dt.total_seconds().dropna()
        if len(time_diffs) > 0:
            features.extend([
                time_diffs.mean(),
                time_diffs.std()
            ])
        else:
            features.extend([0, 0])
        
        return np.array(features)

# Example usage and demonstration
async def ml_pipeline_demo():
    """
    Demonstration of the complete ML pipeline
    Demonstrates: End-to-end ML workflow, model training, evaluation
    """
    import os
    
    logger.info("🤖 Starting ML Pipeline Demo...")
    
    # Create models directory
    os.makedirs('/app/models', exist_ok=True)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 10
    n_features = 5
    n_classes = 15
    
    # Synthetic activity data
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Activity type labels
    activity_types = [
        'sitting', 'standing', 'walking', 'running', 'jumping',
        'hand_flapping', 'rocking', 'spinning', 'head_banging',
        'wandering', 'aggressive_behavior', 'self_harm',
        'focused_activity', 'social_interaction', 'therapy_exercise'
    ]
    
    y_labels = [activity_types[i] for i in y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Flatten sequences for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Initialize models
    rf_model = RandomForestActivityClassifier(n_estimators=100)
    svm_model = SVMActivityClassifier(kernel='rbf')
    
    # Train traditional ML models
    logger.info("Training Random Forest model...")
    rf_model.train(X_train_flat, y_train)
    
    logger.info("Training SVM model...")
    svm_model.train(X_train_flat, y_train)
    
    # Initialize deep learning components
    config = ModelConfig(
        sequence_length=sequence_length,
        input_features=n_features,
        num_classes=n_classes
    )
    
    lstm_model = LSTMActivityClassifier(config)
    trainer = DeepLearningTrainer(lstm_model, config)
    
    # Prepare data for PyTorch
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Create datasets and data loaders
    train_dataset = ActivityDataset(X_train, y_train_encoded)
    test_dataset = ActivityDataset(X_test, y_test_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train deep learning model
    logger.info("Training LSTM model...")
    training_history = trainer.train(train_loader, val_loader)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    
    logger.info("Evaluating models...")
    rf_results = evaluator.evaluate_model(rf_model, X_test_flat, y_test)
    svm_results = evaluator.evaluate_model(svm_model, X_test_flat, y_test)
    
    # Save models
    rf_model.save_model('/app/models/random_forest_model.pkl')
    svm_model.save_model('/app/models/svm_model.pkl')
    torch.save(lstm_model.state_dict(), '/app/models/lstm_model.pth')
    
    # Generate comparison report
    comparison_df = evaluator.compare_models()
    logger.info("Model Comparison Results:")
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Plot confusion matrices
    for model_name in ['RandomForest', 'SVM']:
        evaluator.plot_confusion_matrix(model_name, activity_types)
    
    logger.info("🎉 ML Pipeline Demo completed successfully!")
    return {
        'rf_accuracy': rf_results['accuracy'],
        'svm_accuracy': svm_results['accuracy'],
        'comparison_results': comparison_df.to_dict(),
        'training_history': training_history
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(ml_pipeline_demo())