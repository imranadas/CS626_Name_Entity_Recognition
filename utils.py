# utils.py

import string
from logger_config import setup_logger

logger = setup_logger('utils')

def preprocess_sentence(sentence):
    """
    Preprocess a sentence by lowercasing and tokenizing.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        tuple: (original_tokens, preprocessed_tokens)
    """
    logger.info(f"Preprocessing sentence: {sentence}")
    try:
        original_tokens = sentence.split()
        preprocessed = sentence.lower()
        preprocessed_tokens = preprocessed.split()
        logger.info("Successfully preprocessed sentence")
        return original_tokens, preprocessed_tokens
    except Exception as e:
        logger.error(f"Error preprocessing sentence: {str(e)}")
        raise

def extract_features(tokens, i, preprocessed_tokens):
    """
    Extract features for a given token position.
    
    Args:
        tokens (list): Original tokens
        i (int): Current token position
        preprocessed_tokens (list): Preprocessed tokens
        
    Returns:
        dict: Features dictionary
    """
    try:
        features = {}
        
        # Current token features
        token = tokens[i]
        preprocessed_token = preprocessed_tokens[i]
        
        # Capitalization features
        features['is_capitalized'] = token[0].isupper()
        features['is_all_caps'] = token.isupper()
        features['has_caps_inside'] = any(c.isupper() for c in token[1:])
        
        # Token pattern features
        features['is_alphanumeric'] = any(c.isdigit() for c in token) and any(c.isalpha() for c in token)
        features['has_punctuation'] = any(c in string.punctuation for c in token)
        
        # Length features
        features['token_length'] = len(token)
        features['is_short'] = len(token) <= 3
        
        # Position features
        features['is_first_token'] = i == 0
        features['is_last_token'] = i == len(tokens) - 1
        
        # Surrounding token features
        if i > 0:
            features['prev_token'] = preprocessed_tokens[i-1]
            features['prev_is_cap'] = tokens[i-1][0].isupper()
        else:
            features['prev_token'] = '<START>'
            features['prev_is_cap'] = False
        
        if i < len(tokens) - 1:
            features['next_token'] = preprocessed_tokens[i+1]
            features['next_is_cap'] = tokens[i+1][0].isupper()
        else:
            features['next_token'] = '<END>'
            features['next_is_cap'] = False
        
        # Prefix and suffix features
        if len(preprocessed_token) >= 3:
            features['prefix_3'] = preprocessed_token[:3]
            features['suffix_3'] = preprocessed_token[-3:]
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features for token {token}: {str(e)}")
        raise

def prepare_data(sentences, labels=None):
    """
    Prepare features for a list of sentences.
    
    Args:
        sentences (list): List of input sentences
        labels (list, optional): List of labels
        
    Returns:
        tuple: (features, labels) or just features if no labels provided
    """
    logger.info(f"Preparing features for {len(sentences)} sentences")
    try:
        all_features = []
        all_labels = []
        
        for idx, sentence in enumerate(sentences):
            original_tokens, preprocessed_tokens = preprocess_sentence(sentence)
            
            for i in range(len(original_tokens)):
                features = extract_features(original_tokens, i, preprocessed_tokens)
                all_features.append(features)
                
                if labels is not None and idx < len(labels):
                    all_labels.append(labels[idx])
        
        logger.info(f"Successfully prepared features for {len(all_features)} tokens")
        return all_features, all_labels if labels is not None else all_features
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def load_conll_data(file_path):
    """
    Load CoNLL-2003 data and convert B-* and I-* tags to 1, rest to 0
    
    Args:
        file_path (str): Path to CoNLL format file
        
    Returns:
        tuple: (sentences, labels)
    """
    logger.info(f"Loading CoNLL data from: {file_path}")
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[1]
                        current_sentence.append(token)
                        current_labels.append(1 if tag.startswith(('B-', 'I-')) else 0)
                elif current_sentence:
                    sentences.append(' '.join(current_sentence))
                    labels.extend(current_labels)
                    current_sentence = []
                    current_labels = []
        
        if current_sentence:
            sentences.append(' '.join(current_sentence))
            labels.extend(current_labels)
        
        logger.info(f"Successfully loaded {len(sentences)} sentences from CoNLL data")
        return sentences, labels
    
    except Exception as e:
        logger.error(f"Error loading CoNLL data: {str(e)}")
        raise
