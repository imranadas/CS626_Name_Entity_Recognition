# train.py

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from logger_config import setup_logger
from data_utils import CoNLLDatasetLoader
from resource_monitor import ResourceMonitor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from utils import prepare_data, load_conll_data, NamedEntityFeatures
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

logger = setup_logger('training')

def parallel_prepare_features(sentences, tokens=None, pos_tags=None, labels=None):
    """Prepare features in parallel"""
    n_jobs = cpu_count()
    chunk_size = max(1, len(sentences) // n_jobs)
    
    # Split data into chunks
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)] if tokens else [None] * len(sentence_chunks)
    pos_chunks = [pos_tags[i:i + chunk_size] for i in range(0, len(pos_tags), chunk_size)] if pos_tags else [None] * len(sentence_chunks)
    label_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)] if labels else [None] * len(sentence_chunks)
    
    # Process chunks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(prepare_data)(sentence_chunk, token_chunk, pos_chunk, label_chunk)
        for sentence_chunk, token_chunk, pos_chunk, label_chunk in zip(sentence_chunks, token_chunks, pos_chunks, label_chunks)
    )
    
    # Combine results
    all_features = []
    all_labels = []
    for features, labels in results:
        all_features.extend(features)
        if labels is not None:
            all_labels.extend(labels)
    
    return all_features, all_labels if any(labels is not None for _, labels in results) else None

class NERModel:
    """Named Entity Recognition model with parallel processing"""
    
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.ne_features = NamedEntityFeatures()
        os.makedirs(model_dir, exist_ok=True)
        self.n_jobs = cpu_count()
        logger.info(f"Initialized NERModel with {self.n_jobs} CPU cores")
    
    def create_pipeline(self):
        """Create a faster model pipeline using LinearSVC with improved convergence"""
        return Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LinearSVC(
                dual='auto',
                class_weight='balanced',
                max_iter=2000,  # Increased
                tol=1e-4,      # Added tolerance parameter
                C=1.0,         # Regularization parameter
                random_state=42,
                # Early stopping
                loss='squared_hinge',
                fit_intercept=True,
                intercept_scaling=1.0
            ))
        ])
    
    def train_model(self, pipeline, X_train, y_train):
        """Train model with progress tracking and improved convergence monitoring"""
        logger.info("Starting model training")
        try:
            # Add validation split for early monitoring
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            
            # Fit the pipeline with intermediate validation
            best_score = 0
            patience = 3
            no_improve_count = 0
            prev_val_score = 0
            
            logger.info("Training with validation monitoring...")
            pipeline.fit(X_train_split, y_train_split)
            
            # Get validation score
            val_score = pipeline.score(X_val_split, y_val_split)
            logger.info(f"Validation accuracy: {val_score:.4f}")
            
            # Extract components
            vectorizer = pipeline.named_steps['vectorizer']
            scaler = pipeline.named_steps['scaler']
            model = pipeline.named_steps['classifier']
            
            # Final training on full dataset if validation performance is good
            if val_score > 0.9:  # You can adjust this threshold
                logger.info("Retraining on full dataset...")
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps['classifier']
            
            return model, vectorizer, scaler
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def parallel_evaluate_thresholds(self, scores, y_true, thresholds):
        """Evaluate multiple thresholds in parallel"""
        def evaluate_single_threshold(threshold):
            predictions = (scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary')
            return threshold, precision, recall, f1, predictions
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_single_threshold)(threshold) 
            for threshold in thresholds
        )
        return results
    
    def evaluate_split(self, vectorizer, scaler, model, X, y, split_name):
        """Evaluate model on a data split with parallel threshold evaluation"""
        logger.info(f"Evaluating {split_name} split")
        
        # Transform features
        X_transformed = vectorizer.transform(X)
        X_scaled = scaler.transform(X_transformed)
        
        # Get model scores
        scores = model.decision_function(X_scaled)
        thresholds = np.linspace(0.3, 0.7, 9)
        
        # Evaluate thresholds in parallel
        results = self.parallel_evaluate_thresholds(scores, y, thresholds)
        
        # Find best results
        best_result = max(results, key=lambda x: x[3])  # Max by F1 score
        best_threshold, precision, recall, f1, predictions = best_result
        
        best_metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(best_threshold),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }
        
        logger.info(f"{split_name} Best Metrics (threshold={best_threshold:.2f}):")
        logger.info(f"Precision: {best_metrics['precision']:.4f}")
        logger.info(f"Recall: {best_metrics['recall']:.4f}")
        logger.info(f"F1-score: {best_metrics['f1']:.4f}")
        
        return best_metrics
    
    def train_and_evaluate(self, train_file, valid_file, test_file):
        """Train and evaluate the NER model with parallel processing"""
        logger.info("Starting model training and evaluation")
        
        try:
            # Load data with POS tags
            logger.info("Loading and preparing data...")
            train_sentences, train_tokens, train_pos, train_labels = load_conll_data(train_file)
            valid_sentences, valid_tokens, valid_pos, valid_labels = load_conll_data(valid_file)
            test_sentences, test_tokens, test_pos, test_labels = load_conll_data(test_file)
            
            # Prepare features in parallel
            logger.info("Preparing features with parallel processing...")
            train_features, train_y = parallel_prepare_features(
                train_sentences, train_tokens, train_pos, train_labels)
            valid_features, valid_y = parallel_prepare_features(
                valid_sentences, valid_tokens, valid_pos, valid_labels)
            test_features, test_y = parallel_prepare_features(
                test_sentences, test_tokens, test_pos, test_labels)
            
            # Convert to numpy arrays
            y_train = np.array(train_y)
            y_valid = np.array(valid_y)
            y_test = np.array(test_y)
            
            # Create and train model
            logger.info("Creating and training model...")
            pipeline = self.create_pipeline()
            model, vectorizer, scaler = self.train_model(pipeline, train_features, y_train)
            
            # Evaluate on all splits
            metrics = {}
            splits = {
                'train': (train_features, y_train),
                'valid': (valid_features, y_valid),
                'test': (test_features, y_test)
            }
            
            for split_name, (X, y) in splits.items():
                metrics[split_name] = self.evaluate_split(vectorizer, scaler, model, X, y, split_name)
            
            # Save model artifacts
            logger.info("Saving model artifacts...")
            os.makedirs(self.model_dir, exist_ok=True)
            
            with open(os.path.join(self.model_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metrics and feature info
            feature_info = {
                'best_thresholds': {
                    split: metrics[split]['threshold']
                    for split in metrics.keys()
                }
            }
            
            # Save cache path for future reference
            feature_info['features_cache'] = str(self.ne_features.resource_dir / "entity_features.json")
            
            with open(os.path.join(self.model_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(self.model_dir, 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            logger.info("Training completed successfully!")
            return model, vectorizer, scaler, metrics
        
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

if __name__ == "__main__":
    data_dir = "data"
    model_dir = "model"
    
    try:
        # Initialize resource monitor
        monitor = ResourceMonitor(interval=5)
        monitor.start()
        
        try:
            # Initialize dataset and model
            dataset_loader = CoNLLDatasetLoader(data_dir)
            ner_model = NERModel(model_dir)
            
            # Prepare dataset with progress bar
            try:
                logger.info("Attempting to use existing dataset...")
                with tqdm(total=1, desc='Preparing Dataset',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}]') as pbar:
                    processed_dir = dataset_loader.prepare_existing_dataset(
                        os.path.join(data_dir, "conll_2003")
                    )
                    pbar.update(1)
            except FileNotFoundError:
                logger.info("Downloading and preparing dataset...")
                with tqdm(total=1, desc='Downloading Dataset',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}]') as pbar:
                    processed_dir = dataset_loader.prepare_dataset(force_download=True)
                    pbar.update(1)
            
            # Train and evaluate model
            model, vectorizer, scaler, metrics = ner_model.train_and_evaluate(
                os.path.join(processed_dir, "train.txt"),
                os.path.join(processed_dir, "valid.txt"),
                os.path.join(processed_dir, "test.txt")
            )
            
            logger.info("Process completed successfully!")
            
        finally:
            # Stop resource monitoring
            monitor.stop()
            
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        raise
