"""
End-to-end RAG predictor that uses:
 - Enhanced HNSW retrieval with hierarchical indexing
 - Ollama LLM for feature extraction
 - Enhanced meta-calibrator for final truth prediction
 - Improved topic classification with confidence scoring
"""
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Tuple, Dict, Any, Optional

from enhanced_truth_meta import predict_truth, predict_truth_with_uncertainty


from enhanced_truth_features import METADATA, CHUNKS, TOPIC_MAPPING
# Extract for backward compatibility
PASSAGES = METADATA.get('chunks', {})
TOPIC_MAP = TOPIC_MAPPING

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise in production
logger = logging.getLogger(__name__)

def get_topic_from_passage_id(pid: int) -> int:
    """Extract topic ID from passage metadata with error handling."""
    try:
        passage_data = PASSAGES.get(str(pid), {})
        
        # Try direct topic_id first (from enhanced indexing)
        if 'topic_id' in passage_data:
            return int(passage_data['topic_id'])
        
        # Fallback to topic_name lookup
        topic_name = passage_data.get('topic_name', '')
        if topic_name and topic_name in TOPIC_MAP:
            return int(TOPIC_MAP[topic_name])
        
        # Return -1 for unknown topics
        return -1
        
    except Exception as e:
        logger.warning(f"Failed to extract topic for passage {pid}: {e}")
        return -1

def majority_vote_topic(pids: list[int], confidence_threshold: float = 0.1) -> Tuple[int, float]:
    """
    Enhanced topic prediction with confidence scoring.
    
    Args:
        pids: List of retrieved passage IDs
        confidence_threshold: Minimum confidence for topic prediction
        
    Returns:
        Tuple of (topic_id, confidence_score)
    """
    if not pids:
        return -1, 0.0
    
    # Extract topics from passages
    topics = []
    for pid in pids:
        topic_id = get_topic_from_passage_id(pid)
        if topic_id != -1:
            topics.append(topic_id)
    
    if not topics:
        return -1, 0.0
    
    # Count topic votes
    topic_counts = Counter(topics)
    total_votes = len(topics)
    
    # Get most common topic
    most_common = topic_counts.most_common(1)
    if not most_common:
        return -1, 0.0
    
    predicted_topic, vote_count = most_common[0]
    
    # Calculate confidence as proportion of votes
    confidence = vote_count / total_votes
    
    # Apply confidence threshold
    if confidence < confidence_threshold:
        logger.debug(f"Topic prediction confidence {confidence:.3f} below threshold {confidence_threshold}")
        return -1, confidence
    
    logger.debug(f"Predicted topic {predicted_topic} with confidence {confidence:.3f}")
    return predicted_topic, confidence

def enhanced_topic_prediction(pids: list[int], statement: str = "") -> Tuple[int, float, Dict[str, Any]]:
    """
    Advanced topic prediction using multiple signals.
    
    Args:
        pids: Retrieved passage IDs
        statement: Original statement for additional context
        
    Returns:
        Tuple of (topic_id, confidence, metadata)
    """
    # Basic majority vote
    topic_id, confidence = majority_vote_topic(pids)
    
    # Collect additional metadata
    topic_distribution = defaultdict(int)
    passage_metadata = []
    
    for pid in pids:
        topic = get_topic_from_passage_id(pid)
        if topic != -1:
            topic_distribution[topic] += 1
            
        # Collect passage metadata for analysis
        passage_data = PASSAGES.get(str(pid), {})
        passage_metadata.append({
            'pid': pid,
            'topic_id': topic,
            'topic_name': passage_data.get('topic_name', 'unknown'),
            'chunk_level': passage_data.get('level', 'standard'),
            'key_terms': passage_data.get('key_terms', [])
        })
    
    # Calculate entropy for topic distribution
    total_votes = sum(topic_distribution.values())
    entropy = 0.0
    if total_votes > 0:
        for count in topic_distribution.values():
            if count > 0:
                p = count / total_votes
                entropy -= p * np.log(p)
    
    metadata = {
        'topic_distribution': dict(topic_distribution),
        'entropy': entropy,
        'num_unique_topics': len(topic_distribution),
        'passage_metadata': passage_metadata,
        'retrieval_quality': len(pids) > 0
    }
    
    return topic_id, confidence, metadata

def predict(statement: str, *, k: int = 4, return_confidence: bool = False, 
           enhanced_topic: bool = True) -> Tuple[int, int] | Tuple[int, int, Dict[str, Any]]:
    """
    Main prediction function for truth and topic classification.
    
    Args:
        statement: Medical statement to classify
        k: Number of passages to retrieve
        return_confidence: Whether to return confidence scores
        enhanced_topic: Whether to use enhanced topic prediction
        
    Returns:
        - If return_confidence=False: (truth_prediction, topic_prediction)  
        - If return_confidence=True: (truth_prediction, topic_prediction, metadata)
    """
    try:
        # Get truth prediction with uncertainty if requested
        if return_confidence:
            truth_pred, truth_confidence, truth_meta = predict_truth_with_uncertainty(statement, k=k)
        else:
            truth_pred, truth_meta = predict_truth(statement, k=k)
            truth_confidence = truth_meta.get('confidence', 0.5)
        
        # Extract retrieved passage IDs
        pids = truth_meta.get("pids", [])
        
        # Topic prediction
        if enhanced_topic:
            topic_pred, topic_confidence, topic_meta = enhanced_topic_prediction(pids, statement)
        else:
            # Legacy topic prediction
            topic_pred, topic_confidence = majority_vote_topic(pids)
            topic_meta = {}
        
        logger.debug(f"Truth: {truth_pred} (conf: {truth_confidence:.3f}), "
                    f"Topic: {topic_pred} (conf: {topic_confidence:.3f})")
        
        if return_confidence:
            # Return comprehensive metadata
            full_metadata = {
                'truth': {
                    'prediction': truth_pred,
                    'confidence': truth_confidence,
                    'features': truth_meta.get('features', {}),
                    'model_type': truth_meta.get('model_type', 'unknown')
                },
                'topic': {
                    'prediction': topic_pred,
                    'confidence': topic_confidence,
                    'metadata': topic_meta
                },
                'retrieval': {
                    'passage_ids': pids,
                    'num_passages': len(pids),
                    'k_requested': k
                },
                'statement_info': {
                    'length': len(statement),
                    'word_count': len(statement.split())
                }
            }
            return truth_pred, topic_pred, full_metadata
        else:
            # Simple return for backward compatibility
            return truth_pred, topic_pred
            
    except Exception as e:
        logger.error(f"Prediction failed for statement: {statement[:100]}... Error: {e}")
        
        # Return conservative fallback predictions
        if return_confidence:
            error_metadata = {
                'error': str(e),
                'truth': {'prediction': 0, 'confidence': 0.0},
                'topic': {'prediction': -1, 'confidence': 0.0},
                'retrieval': {'passage_ids': [], 'num_passages': 0, 'k_requested': k}
            }
            return 0, -1, error_metadata
        else:
            return 0, -1

# Legacy compatibility function
def _majority_topic(pids: list[int]) -> int:
    """Legacy function for backward compatibility."""
    topic_pred, _ = majority_vote_topic(pids)
    return topic_pred

# Batch prediction support
def predict_batch(statements: list[str], *, k: int = 4, return_confidence: bool = False) -> list:
    """
    Batch prediction for multiple statements.
    
    Args:
        statements: List of medical statements
        k: Number of passages to retrieve per statement
        return_confidence: Whether to return confidence scores
        
    Returns:
        List of prediction tuples
    """
    results = []
    
    logger.info(f"Processing {len(statements)} statements in batch")
    
    for i, statement in enumerate(statements):
        try:
            result = predict(statement, k=k, return_confidence=return_confidence)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(statements)} statements")
                
        except Exception as e:
            logger.error(f"Failed to process statement {i}: {e}")
            if return_confidence:
                error_result = (0, -1, {'error': str(e)})
            else:
                error_result = (0, -1)
            results.append(error_result)
    
    return results

# System health check
def system_health_check() -> Dict[str, Any]:
    """Check if the entire RAG system is working correctly."""
    try:
        # Import health check from truth_meta
        from enhanced_truth_meta import model_health_check
        
        model_status = model_health_check()
        
        # Test full system with a simple prediction
        test_statement = "Neonatal testicular torsion typically occurs as extravaginal torsion because the tunica vaginalis has not adhered to the gubernaculum, allowing both the tunica vaginalis and spermatic cord to twist together."
        truth_pred, topic_pred, metadata = predict(test_statement, k=3, return_confidence=True)
        
        system_status = {
            'status': 'healthy',
            'model_health': model_status,
            'retrieval_system': {
                'passages_loaded': len(PASSAGES) > 0,
                'topic_mapping_loaded': len(TOPIC_MAP) > 0,
                'num_passages': len(PASSAGES),
                'num_topics': len(TOPIC_MAP)
            },
            'test_prediction': {
                'truth': truth_pred,
                'topic': topic_pred,
                'truth_confidence': metadata['truth']['confidence'],
                'topic_confidence': metadata['topic']['confidence'],
                'passages_retrieved': metadata['retrieval']['num_passages']
            }
        }
        
        return system_status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'retrieval_system': {
                'passages_loaded': len(PASSAGES) > 0,
                'topic_mapping_loaded': len(TOPIC_MAP) > 0
            }
        }

# Add numpy import for entropy calculation
try:
    import numpy as np
except ImportError:
    # Fallback entropy calculation without numpy
    import math
    
    class np:
        @staticmethod
        def log(x):
            return math.log(x)