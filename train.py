# train.py

import os
import json
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from logger_config import setup_logger
from data_utils import CoNLLDatasetLoader
from utils import prepare_data, load_conll_data
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

logger = setup_logger('training')

def train_and_evaluate(train_file, valid_file, test_file, model_output_dir):
    """
    Train SVM model and evaluate on all splits
    
    Args:
        train_file (str): Path to training file
        valid_file (str): Path to validation file
        test_file (str): Path to test file
        model_output_dir (str): Directory to save model artifacts
        
    Returns:
        tuple: (model, vectorizer, metrics)
    """
    logger.info("Starting model training and evaluation")
    os.makedirs(model_output_dir, exist_ok=True)
    
    try:
        # Load and prepare all data
        logger.info("Loading and preparing data...")
        train_sentences, train_labels = load_conll_data(train_file)
        valid_sentences, valid_labels = load_conll_data(valid_file)
        test_sentences, test_labels = load_conll_data(test_file)
        
        logger.info("Preparing features...")
        train_features, _ = prepare_data(train_sentences)
        valid_features, _ = prepare_data(valid_sentences)
        test_features, _ = prepare_data(test_sentences)
        
        # Vectorize features
        logger.info("Vectorizing features...")
        vectorizer = DictVectorizer()
        X_train = vectorizer.fit_transform(train_features)
        X_valid = vectorizer.transform(valid_features)
        X_test = vectorizer.transform(test_features)
        
        # Convert labels to numpy arrays
        y_train = np.array(train_labels)
        y_valid = np.array(valid_labels)
        y_test = np.array(test_labels)
        
        # Train model
        logger.info("Training SVM model...")
        model = LinearSVC(random_state=42, max_iter=2000)
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Calculate metrics for all splits
        metrics = {}
        splits = {
            'train': (X_train, y_train),
            'valid': (X_valid, y_valid),
            'test': (X_test, y_test)
        }
        
        logger.info("Calculating metrics for all splits...")
        for split_name, (X, y) in splits.items():
            y_pred = model.predict(X)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            conf_matrix = confusion_matrix(y, y_pred).tolist()
            
            metrics[split_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': conf_matrix
            }
            
            logger.info(f"{split_name.capitalize()} Split Metrics:")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-score: {f1:.4f}")
        
        # Save model artifacts
        logger.info("Saving model artifacts...")
        with open(os.path.join(model_output_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(model_output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(model_output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training and evaluation completed successfully!")
        return model, vectorizer, metrics
        
    except Exception as e:
        logger.error(f"Error during training and evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    data_dir = "data"
    model_dir = "model"
    
    try:
        # Initialize dataset loader
        logger.info("Initializing dataset loader...")
        dataset_loader = CoNLLDatasetLoader(data_dir)
        
        # Try to prepare existing dataset first
        try:
            logger.info("Attempting to use existing dataset...")
            processed_dir = dataset_loader.prepare_existing_dataset(
                os.path.join(data_dir, "conll_2003")
            )
        except FileNotFoundError as e:
            logger.warning(f"Existing dataset not found: {str(e)}")
            logger.info("Downloading and preparing dataset...")
            processed_dir = dataset_loader.prepare_dataset(force_download=True)
        
        # Get file paths
        train_file = os.path.join(processed_dir, "train.txt")
        valid_file = os.path.join(processed_dir, "valid.txt")
        test_file = os.path.join(processed_dir, "test.txt")
        
        # Train and evaluate model
        logger.info("Starting training process...")
        model, vectorizer, metrics = train_and_evaluate(
            train_file, valid_file, test_file, model_dir
        )
        
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        logger.error("Please ensure the dataset is available or can be downloaded.")
        raise
