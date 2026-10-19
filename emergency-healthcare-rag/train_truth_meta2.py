"""
Enhanced training script for Truth Meta-Calibrator with comprehensive features.

Trains multiple model variants with advanced feature engineering:
- LogisticRegression with feature selection
- Gradient Boosting for non-linear patterns
- Ensemble methods with uncertainty quantification
- Cross-validation with medical domain stratification

Usage:
    python train_enhanced_truth_meta.py --k 8 --models lr,gbm,ensemble --cv-folds 5
"""

import json, os, argparse, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# Updated import for enhanced features
from enhanced_truth_features import compute_enhanced_features, compute_features_batch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
STATEMENTS_DIR = Path("data/train/statements")
ANSWERS_DIR = Path("data/train/answers")
VAL_STATEMENTS_DIR = Path("data/validation/statements")
VAL_ANSWERS_DIR = Path("data/validation/answers")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Enhanced feature names based on the improved feature extractor
ENHANCED_FEATURE_NAMES = [
    # Core LLM features
    "llm_verdict", "llm_confidence", "has_reasoning", "reasoning_length", "key_evidence_count",
    
    # Similarity features
    "sim_mean", "sim_max", "sim_min", "sim_std", "sim_gap_top1_top2", 
    "sim_gap_top1_median", "sim_quartile_range",
    
    # Topic analysis
    "vote_entropy", "unique_topics", "dominant_topic_ratio", "topic_agreement",
    
    # Hierarchical chunk features  
    "level_context", "level_standard", "level_fact", "level_diversity",
    
    # Medical domain features
    "domain_cardiology", "domain_pulmonology", "domain_neurology", 
    "domain_emergency", "domain_pharmacology", "primary_domain_score",
    
    # Quantifier and uncertainty
    "quantifier_universal", "quantifier_majority", "quantifier_minority", "quantifier_precise",
    "uncertainty_markers", "negation_count", "negation_early", "negation_spread",
    
    # Semantic coherence
    "semantic_coherence", "max_coherence", "coherence_std",
    
    # Key term alignment
    "key_term_overlap", "key_term_density", "unique_key_terms",
    
    # Surface features
    "stmt_len_words", "stmt_len_chars", "avg_word_length", "has_percent", 
    "has_numbers", "num_digits", "num_punctuation", "sentence_count"
]

def load_training_pairs() -> List[Tuple[str, int, str]]:
    """Load training pairs with metadata for stratification."""
    pairs = []
    
    # Build label mapping
    labels: Dict[str, Dict[str, Any]] = {}
    for answer_file in ANSWERS_DIR.glob("*.json"):
        stem = answer_file.stem
        try:
            answer_data = json.loads(answer_file.read_text())
            labels[stem] = {
                'truth': int(answer_data["statement_is_true"]),
                'topic': answer_data.get("statement_topic", -1)
            }
        except Exception as e:
            logger.warning(f"Failed to parse {answer_file}: {e}")
            continue
    
    # Collect statements with labels
    for statement_file in STATEMENTS_DIR.glob("*.txt"):
        stem = statement_file.stem
        if stem in labels:
            try:
                statement_text = statement_file.read_text().strip()
                if statement_text:  # Skip empty statements
                    pairs.append((
                        statement_text, 
                        labels[stem]['truth'], 
                        str(labels[stem].get('topic', -1))
                    ))
            except Exception as e:
                logger.warning(f"Failed to read {statement_file}: {e}")
                continue
    
    logger.info(f"Loaded {len(pairs)} training pairs")
    return pairs

def load_validation_pairs() -> Optional[List[Tuple[str, int, str]]]:
    """Load validation pairs if available."""
    if not VAL_STATEMENTS_DIR.exists() or not VAL_ANSWERS_DIR.exists():
        logger.info("Validation data not found, skipping validation set evaluation")
        return None
    
    pairs = []
    
    # Build label mapping for validation
    labels: Dict[str, Dict[str, Any]] = {}
    for answer_file in VAL_ANSWERS_DIR.glob("*.json"):
        stem = answer_file.stem
        try:
            answer_data = json.loads(answer_file.read_text())
            labels[stem] = {
                'truth': int(answer_data["statement_is_true"]),
                'topic': answer_data.get("statement_topic", -1)
            }
        except Exception as e:
            logger.warning(f"Failed to parse validation {answer_file}: {e}")
            continue
    
    # Collect validation statements
    for statement_file in VAL_STATEMENTS_DIR.glob("*.txt"):
        stem = statement_file.stem
        if stem in labels:
            try:
                statement_text = statement_file.read_text().strip()
                if statement_text:
                    pairs.append((
                        statement_text,
                        labels[stem]['truth'],
                        str(labels[stem].get('topic', -1))
                    ))
            except Exception as e:
                logger.warning(f"Failed to read validation {statement_file}: {e}")
                continue
    
    logger.info(f"Loaded {len(pairs)} validation pairs")
    return pairs

def extract_features_batch(statements: List[str], k: int = 8) -> Tuple[np.ndarray, List[str]]:
    """Extract features from statements in batch with error handling."""
    logger.info(f"Extracting features from {len(statements)} statements with k={k}")
    
    # Use batch processing for efficiency
    results = compute_features_batch(statements, k=k)
    
    feature_vectors = []
    missing_features = set()
    
    for i, (features, _) in enumerate(results):
        # Ensure all expected features are present
        feature_vector = []
        for feature_name in ENHANCED_FEATURE_NAMES:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                feature_vector.append(0.0)  # Default value for missing features
                missing_features.add(feature_name)
        
        feature_vectors.append(feature_vector)
    
    if missing_features:
        logger.warning(f"Missing features in some statements: {missing_features}")
    
    X = np.array(feature_vectors, dtype=np.float32)
    
    # Handle any remaining NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    logger.info(f"Extracted feature matrix of shape {X.shape}")
    return X, ENHANCED_FEATURE_NAMES

def create_domain_stratified_splits(pairs: List[Tuple[str, int, str]], n_splits: int = 5) -> StratifiedKFold:
    """Create stratified splits considering both class balance and topic diversity."""
    # Create stratification key combining truth label and topic
    y = np.array([pair[1] for pair in pairs])  # Truth labels
    topics = np.array([pair[2] for pair in pairs])  # Topic IDs
    
    # For stratification, we'll use just the truth labels but log topic distribution
    topic_counts = pd.Series(topics).value_counts()
    logger.info(f"Topic distribution: {len(topic_counts)} unique topics")
    logger.info(f"Truth label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def perform_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                            method: str = 'rfe') -> Tuple[np.ndarray, List[str], Any]:
    """Perform feature selection using specified method."""
    logger.info(f"Performing feature selection using method: {method}")
    
    if method == 'kbest':
        # Select K best features using F-score
        k = min(25, X.shape[1])  # Select top 25 or all features if fewer
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
    elif method == 'rfe':
        # Recursive Feature Elimination with cross-validation
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFECV(estimator, step=1, cv=3, scoring='accuracy', n_jobs=-1)
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        
    else:  # 'none' - use all features
        X_selected = X
        selected_features = feature_names
        selector = None
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
    return X_selected, selected_features, selector

def train_logistic_regression(X: np.ndarray, y: np.ndarray, cv_folds: StratifiedKFold) -> Dict[str, Any]:
    """Train and tune Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  # Supports both L1 and L2
        'class_weight': ['balanced', None]
    }
    
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_folds, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    logger.info(f"Best LR parameters: {grid_search.best_params_}")
    logger.info(f"Best LR CV score: {grid_search.best_score_:.4f}")
    
    return {
        'model': grid_search.best_estimator_,
        'cv_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'grid_search': grid_search
    }

def train_gradient_boosting(X: np.ndarray, y: np.ndarray, cv_folds: StratifiedKFold) -> Dict[str, Any]:
    """Train and tune Gradient Boosting model."""
    logger.info("Training Gradient Boosting model...")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sample_weights = np.array([class_weights[label] for label in y])
    
    # Hyperparameter grid (smaller for faster training)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
    }
    
    base_model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_folds, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    # Fit with sample weights to handle class imbalance
    grid_search.fit(X, y, sample_weight=sample_weights)
    
    logger.info(f"Best GBM parameters: {grid_search.best_params_}")
    logger.info(f"Best GBM CV score: {grid_search.best_score_:.4f}")
    
    return {
        'model': grid_search.best_estimator_,
        'cv_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'grid_search': grid_search
    }

def create_ensemble_model(models: Dict[str, Any]) -> VotingClassifier:
    """Create ensemble model from trained models."""
    logger.info("Creating ensemble model...")
    
    estimators = []
    for name, model_info in models.items():
        if 'model' in model_info:
            estimators.append((name, model_info['model']))
    
    # Use soft voting for probability-based predictions
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    return ensemble

def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, cv_folds: StratifiedKFold, 
                  model_name: str) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    logger.info(f"Evaluating {model_name}...")
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    cv_precision = cross_val_score(model, X, y, cv=cv_folds, scoring='precision', n_jobs=-1)
    cv_recall = cross_val_score(model, X, y, cv=cv_folds, scoring='recall', n_jobs=-1)
    cv_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1', n_jobs=-1)
    cv_auc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    metrics = {
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_precision_mean': cv_precision.mean(),
        'cv_precision_std': cv_precision.std(),
        'cv_recall_mean': cv_recall.mean(),
        'cv_recall_std': cv_recall.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
    }
    
    logger.info(f"{model_name} CV Results:")
    logger.info(f"  Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    logger.info(f"  Precision: {metrics['cv_precision_mean']:.4f} ± {metrics['cv_precision_std']:.4f}")
    logger.info(f"  Recall: {metrics['cv_recall_mean']:.4f} ± {metrics['cv_recall_std']:.4f}")
    logger.info(f"  F1-Score: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    
    return metrics

def save_model_artifacts(models: Dict[str, Any], scaler: Any, feature_names: List[str], 
                        selector: Any, metrics: Dict[str, Dict[str, float]], 
                        X_selected: np.ndarray, y: np.ndarray,
                        model_prefix: str = "enhanced_truth_meta"):
    """Save all model artifacts and metadata."""
    
    # Save individual models
    for model_name, model_info in models.items():
        if 'model' in model_info:
            model_path = OUT_DIR / f"{model_prefix}_{model_name}.joblib"
            
            artifact = {
                'model': model_info['model'],
                'scaler': scaler,
                'feature_names': feature_names,
                'selector': selector,
                'metrics': metrics.get(model_name, {}),
                'best_params': model_info.get('best_params', {}),
                'model_type': model_name
            }
            
            dump(artifact, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save ensemble if multiple models exist
    if len(models) > 1:
        # Create and train ensemble on the already processed data
        ensemble = create_ensemble_model(models)
        ensemble.fit(X_selected, y)  # Use X_selected which is already scaled and feature-selected
        
        ensemble_path = OUT_DIR / f"{model_prefix}_ensemble.joblib"
        ensemble_artifact = {
            'model': ensemble,
            'scaler': scaler,
            'feature_names': feature_names,  # These are already the selected feature names
            'selector': selector,
            'component_models': list(models.keys()),
            'model_type': 'ensemble'
        }
        
        dump(ensemble_artifact, ensemble_path)
        logger.info(f"Saved ensemble model to {ensemble_path}")
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'models': list(models.keys()),
        'metrics': metrics,
        'training_timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = OUT_DIR / f"{model_prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved training metadata to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Truth Meta-Calibrator Training")
    parser.add_argument("--k", type=int, default=8, help="Number of passages to retrieve")
    parser.add_argument("--models", type=str, default="lr,gbm", 
                       help="Comma-separated list of models: lr,gbm,ensemble")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--feature-selection", type=str, default="rfe", 
                       choices=["none", "kbest", "rfe"], help="Feature selection method")
    parser.add_argument("--scaler", type=str, default="standard", 
                       choices=["standard", "robust"], help="Feature scaler")
    parser.add_argument("--validate", action="store_true", help="Evaluate on validation set")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Enhanced Truth Meta-Calibrator Training")
    logger.info(f"Configuration: k={args.k}, models={args.models}, cv_folds={args.cv_folds}")
    
    # Load training data
    train_pairs = load_training_pairs()
    if not train_pairs:
        logger.error("No training pairs found!")
        return
    
    # Extract statements and labels
    statements = [pair[0] for pair in train_pairs]
    y = np.array([pair[1] for pair in train_pairs], dtype=np.int32)
    
    # Extract features
    X, feature_names = extract_features_batch(statements, k=args.k)
    
    logger.info(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Feature scaling
    if args.scaler == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    X_selected, selected_features, selector = perform_feature_selection(
        X_scaled, y, feature_names, method=args.feature_selection
    )
    
    # Create stratified CV folds
    cv_folds = create_domain_stratified_splits(train_pairs, n_splits=args.cv_folds)
    
    # Train models
    models = {}
    model_list = args.models.split(',')
    
    if 'lr' in model_list:
        models['logistic_regression'] = train_logistic_regression(X_selected, y, cv_folds)
    
    if 'gbm' in model_list:
        models['gradient_boosting'] = train_gradient_boosting(X_selected, y, cv_folds)
    
    # Evaluate all models
    metrics = {}
    for model_name, model_info in models.items():
        if 'model' in model_info:
            model_metrics = evaluate_model(
                model_info['model'], X_selected, y, cv_folds, model_name
            )
            metrics[model_name] = model_metrics
    
    # Evaluate ensemble if multiple models
    if len(models) > 1 and 'ensemble' in model_list:
        ensemble = create_ensemble_model(models)
        ensemble.fit(X_selected, y)
        ensemble_metrics = evaluate_model(ensemble, X_selected, y, cv_folds, 'ensemble')
        metrics['ensemble'] = ensemble_metrics
        models['ensemble'] = {'model': ensemble}
    
    # Validation set evaluation
    if args.validate:
        val_pairs = load_validation_pairs()
        if val_pairs:
            val_statements = [pair[0] for pair in val_pairs]
            y_val = np.array([pair[1] for pair in val_pairs], dtype=np.int32)
            
            X_val, _ = extract_features_batch(val_statements, k=args.k)
            X_val_scaled = scaler.transform(X_val)
            X_val_selected = selector.transform(X_val_scaled) if selector else X_val_scaled
            
            logger.info("🔍 Validation Set Results:")
            for model_name, model_info in models.items():
                if 'model' in model_info:
                    val_pred = model_info['model'].predict(X_val_selected)
                    val_accuracy = accuracy_score(y_val, val_pred)
                    val_f1 = f1_score(y_val, val_pred)
                    logger.info(f"  {model_name}: Accuracy={val_accuracy:.4f}, F1={val_f1:.4f}")
    
    # Save models and artifacts - pass the correctly processed data
    save_model_artifacts(models, scaler, selected_features, selector, metrics, X_selected, y)
    
    # Print summary
    logger.info("📊 Training Summary:")
    best_model = max(metrics.items(), key=lambda x: x[1]['cv_accuracy_mean'])
    logger.info(f"Best model: {best_model[0]} (CV Accuracy: {best_model[1]['cv_accuracy_mean']:.4f})")
    
    logger.info("✅ Enhanced Truth Meta-Calibrator training completed successfully!")

if __name__ == "__main__":
    main()