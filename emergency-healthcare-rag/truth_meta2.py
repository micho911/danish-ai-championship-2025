"""
Runtime helper: load the trained meta-calibrator and predict truth.
Updated to work with enhanced models while maintaining backward compatibility.
"""
import numpy as np
import logging
from joblib import load
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from enhanced_truth_features import compute_enhanced_features, compute_features

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise in production
logger = logging.getLogger(__name__)

# Model loading with fallback strategy
MODEL_PATHS = [
    "models/enhanced_truth_meta_logistic_regression.joblib",  # New enhanced model
    "models/enhanced_truth_meta_ensemble.joblib",            # Ensemble model
    "models/truth_meta2.joblib",                             # Legacy model
    "models/truth_meta.joblib"                               # Original model
]

def load_best_available_model() -> Tuple[Any, Any, Any, list, str, Optional[Any]]:
    """Load the best available model with fallback to legacy versions."""
    
    for model_path in MODEL_PATHS:
        if Path(model_path).exists():
            try:
                logger.info(f"Loading model from {model_path}")
                model_bundle = load(model_path)
                
                # Extract components based on model structure
                scaler = model_bundle["scaler"]
                model = model_bundle["model"]
                feature_names = model_bundle["feature_names"]
                model_type = model_bundle.get("model_type", "legacy")
                selector = model_bundle.get("selector", None)
                
                logger.info(f"Successfully loaded {model_type} model with {len(feature_names)} features")
                return scaler, model, feature_names, model_type, selector, model_path
                
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {e}")
                continue
    
    raise FileNotFoundError(f"No valid model found in any of: {MODEL_PATHS}")

# Load model at module import
try:
    SCALER, MODEL, FEAT_NAMES, MODEL_TYPE, SELECTOR, MODEL_PATH = load_best_available_model()
    logger.info(f"Initialized with {MODEL_TYPE} model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load any model: {e}")
    raise

# Legacy feature names for backward compatibility
LEGACY_FEATURE_NAMES = [
    "llm_verdict",
    "sim_mean", "sim_max", "sim_min", "sim_std", "sim_gap_top1_top2",
    "vote_entropy", "unique_topics",
    "stmt_len_words", "stmt_len_chars",
    "neg_count", "has_percent", "num_digits",
]

def get_default_feature_values() -> Dict[str, float]:
    """Get default values for all expected features."""
    defaults = {}
    
    # Enhanced features default values
    enhanced_defaults = {
        "llm_verdict": 0.0,
        "llm_confidence": 0.5,
        "has_reasoning": 0.0,
        "reasoning_length": 0.0,
        "key_evidence_count": 0.0,
        
        # Similarity features
        "sim_mean": 0.0, "sim_max": 0.0, "sim_min": 0.0, "sim_std": 0.0,
        "sim_gap_top1_top2": 0.0, "sim_gap_top1_median": 0.0, "sim_quartile_range": 0.0,
        
        # Topic features
        "vote_entropy": 0.0, "unique_topics": 0.0, "dominant_topic_ratio": 0.0, "topic_agreement": 0.0,
        
        # Level features
        "level_context": 0.0, "level_standard": 0.0, "level_fact": 0.0, "level_diversity": 0.0,
        
        # Medical domain features
        "domain_cardiology": 0.0, "domain_pulmonology": 0.0, "domain_neurology": 0.0, 
        "domain_emergency": 0.0, "domain_pharmacology": 0.0, "primary_domain_score": 0.0,
        
        # Quantifier features
        "quantifier_universal": 0.0, "quantifier_majority": 0.0, "quantifier_minority": 0.0, "quantifier_precise": 0.0,
        "uncertainty_markers": 0.0, "negation_count": 0.0, "negation_early": 0.0, "negation_spread": 0.0,
        
        # Coherence features
        "semantic_coherence": 0.0, "max_coherence": 0.0, "coherence_std": 0.0,
        
        # Alignment features
        "key_term_overlap": 0.0, "key_term_density": 0.0, "unique_key_terms": 0.0,
        
        # Surface features
        "stmt_len_words": 0.0, "stmt_len_chars": 0.0, "avg_word_length": 0.0,
        "has_percent": 0.0, "has_numbers": 0.0, "num_digits": 0.0, 
        "num_punctuation": 0.0, "sentence_count": 1.0,
        
        # Legacy compatibility
        "neg_count": 0.0
    }
    
    return enhanced_defaults

def extract_legacy_features(enhanced_features: Dict[str, float]) -> Dict[str, float]:
    """Extract legacy features from enhanced feature set for backward compatibility."""
    legacy_features = {}
    
    for feat_name in LEGACY_FEATURE_NAMES:
        if feat_name in enhanced_features:
            legacy_features[feat_name] = enhanced_features[feat_name]
        else:
            # Map enhanced features to legacy names where possible
            if feat_name == "neg_count":
                legacy_features[feat_name] = enhanced_features.get("negation_count", 0.0)
            else:
                # Provide sensible defaults for missing legacy features
                defaults = {
                    "llm_verdict": 0.0,
                    "sim_mean": 0.0, "sim_max": 0.0, "sim_min": 0.0, "sim_std": 0.0,
                    "sim_gap_top1_top2": 0.0, "vote_entropy": 0.0, "unique_topics": 0.0,
                    "stmt_len_words": 0.0, "stmt_len_chars": 0.0,
                    "has_percent": 0.0, "num_digits": 0.0
                }
                legacy_features[feat_name] = defaults.get(feat_name, 0.0)
    
    return legacy_features

def features_to_vector(features: Dict[str, float], feature_names: list) -> np.ndarray:
    """Convert feature dictionary to vector array with proper error handling."""
    try:
        # Ensure we have all required features
        if len(features) == 0:
            logger.warning("Empty features dictionary, using defaults")
            default_features = get_default_feature_values()
            feature_vector = np.array([default_features.get(name, 0.0) for name in feature_names], dtype="float32")[None, :]
        else:
            feature_vector = np.array([features.get(name, 0.0) for name in feature_names], dtype="float32")[None, :]
        
        # Verify dimensions
        if feature_vector.shape[1] != len(feature_names):
            logger.error(f"Feature vector shape mismatch: got {feature_vector.shape[1]}, expected {len(feature_names)}")
            # Create properly sized fallback vector
            feature_vector = np.zeros((1, len(feature_names)), dtype="float32")
        
        logger.debug(f"Created feature vector with shape {feature_vector.shape}")
        return feature_vector
        
    except Exception as e:
        logger.error(f"Feature vectorization failed: {e}")
        # Return properly sized zero vector as fallback
        logger.info(f"Creating fallback zero vector with {len(feature_names)} features")
        return np.zeros((1, len(feature_names)), dtype="float32")

def predict_truth(statement: str, k: int = 4) -> Tuple[int, dict]:
    """
    Predict truth of a medical statement using the loaded meta-calibrator.
    
    Args:
        statement: Medical statement to evaluate
        k: Number of passages to retrieve for context
        
    Returns:
        Tuple of (prediction, metadata_dict)
        - prediction: 0 (false) or 1 (true)
        - metadata: Dictionary containing features and retrieved passage IDs
    """
    try:
        logger.debug(f"Predicting truth for: {statement[:100]}...")
        
        # Use enhanced feature extraction if available and model supports it
        if MODEL_TYPE != "legacy" and len(FEAT_NAMES) > len(LEGACY_FEATURE_NAMES):
            logger.debug("Using enhanced feature extraction")
            try:
                features, pids = compute_enhanced_features(statement, k=k)
                logger.debug(f"Enhanced features extracted: {len(features)} features")
            except Exception as e:
                logger.error(f"Enhanced feature extraction failed: {e}")
                # Fallback to default features
                features = get_default_feature_values()
                pids = []
        else:
            logger.debug("Using legacy feature extraction")
            try:
                # Fall back to legacy compute_features for older models
                features, pids = compute_features(statement, k=k)
                logger.debug(f"Legacy features extracted: {len(features)} features")
            except Exception as e:
                logger.error(f"Legacy feature extraction failed: {e}")
                # Create minimal feature set
                features = extract_legacy_features(get_default_feature_values())
                pids = []
            
        # Handle feature compatibility
        if MODEL_TYPE == "legacy" or len(FEAT_NAMES) == len(LEGACY_FEATURE_NAMES):
            # Legacy model - extract only legacy features
            features = extract_legacy_features(features)
            logger.debug(f"Using legacy features: {len(features)} features")
        
        # Debug: Check feature completeness
        missing_features = [name for name in FEAT_NAMES if name not in features]
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")  # Show first 10
            # Add missing features with default values
            defaults = get_default_feature_values()
            for name in missing_features:
                features[name] = defaults.get(name, 0.0)
        
        logger.debug(f"Final feature count: {len(features)}, Expected: {len(FEAT_NAMES)}")
        
        # Convert features to vector
        feature_vector = features_to_vector(features, FEAT_NAMES)
        logger.debug(f"Feature vector shape: {feature_vector.shape}")
        
        # Apply feature scaling
        try:
            scaled_features = SCALER.transform(feature_vector)
            logger.debug(f"Scaled features shape: {scaled_features.shape}")
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            logger.error(f"Feature vector shape: {feature_vector.shape}, Expected features: {len(FEAT_NAMES)}")
            # Use unscaled as fallback but log the issue
            scaled_features = feature_vector
        
        # Apply feature selection if available
        if SELECTOR is not None:
            try:
                scaled_features = SELECTOR.transform(scaled_features)
                logger.debug(f"Selected features shape: {scaled_features.shape}")
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")
                # Continue with unselected features
        
        # Make prediction
        try:
            prediction = int(MODEL.predict(scaled_features)[0])
            logger.debug(f"Model prediction: {prediction}")
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            prediction = 0  # Conservative fallback
        
        # Get prediction confidence if available
        confidence = None
        try:
            if hasattr(MODEL, 'predict_proba'):
                proba = MODEL.predict_proba(scaled_features)[0]
                confidence = float(max(proba))  # Confidence as max probability
                logger.debug(f"Prediction confidence: {confidence}")
        except Exception as e:
            logger.debug(f"Confidence extraction failed: {e}")
        
        # Prepare metadata
        metadata = {
            "features": dict(features),  # Ensure it's a regular dict
            "pids": pids,
            "model_type": MODEL_TYPE,
            "model_path": MODEL_PATH,
            "confidence": confidence,
            "num_features_used": len(FEAT_NAMES),
            "feature_vector_shape": feature_vector.shape
        }
        
        logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        return prediction, metadata
        
    except Exception as e:
        logger.error(f"predict_truth failed: {e}")
        # Return conservative fallback
        return 0, {
            "features": {}, 
            "pids": [], 
            "error": str(e),
            "model_type": MODEL_TYPE
        }

def predict_truth_with_uncertainty(statement: str, k: int = 4) -> Tuple[int, float, dict]:
    """
    Extended prediction function that returns uncertainty estimate.
    
    Args:
        statement: Medical statement to evaluate  
        k: Number of passages to retrieve
        
    Returns:
        Tuple of (prediction, confidence, metadata)
    """
    prediction, metadata = predict_truth(statement, k=k)
    confidence = metadata.get("confidence", 0.5)  # Default to neutral confidence
    
    return prediction, confidence, metadata

# Backward compatibility functions
def _vec(feats: dict) -> np.ndarray:
    """Legacy function for backward compatibility."""
    return features_to_vector(feats, FEAT_NAMES)

# Health check function
def model_health_check() -> Dict[str, Any]:
    """Check if the model is loaded and working correctly."""
    try:
        # Test with a simple statement
        test_statement = "NSAID-induced peptic ulcers result from inhibition of cyclooxygenase-2 (COX-2) enzymes, which leads to decreased production of protective prostaglandins and subsequent mucosal damage in the gastroduodenal tract."
        pred, meta = predict_truth(test_statement, k=2)
        
        return {
            "status": "healthy",
            "model_type": MODEL_TYPE,
            "model_path": MODEL_PATH,
            "num_features": len(FEAT_NAMES),
            "test_prediction": pred,
            "test_confidence": meta.get("confidence"),
            "feature_extraction": len(meta.get("features", {})) > 0,
            "retrieval": len(meta.get("pids", [])) > 0,
            "feature_names_sample": FEAT_NAMES[:10] if len(FEAT_NAMES) > 10 else FEAT_NAMES
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_type": MODEL_TYPE if 'MODEL_TYPE' in globals() else "unknown"
        }