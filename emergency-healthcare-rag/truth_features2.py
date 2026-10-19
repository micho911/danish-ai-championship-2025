"""
Enhanced Feature extractor for the Truth Meta-Calibrator.
Computes comprehensive features leveraging the improved hierarchical indexing:
  - llm_verdict (0/1) from Ollama with multi-level context
  - retrieval similarities across different chunk levels
  - topic vote entropy and confidence metrics
  - medical-specific linguistic features
  - key term alignment and semantic coherence
  - uncertainty quantification features
"""

import json, math, os, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import statistics

import numpy as np
import hnswlib
import ollama
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
INDEX_PATH = Path("data/index_enhanced.hnsw")
META_PATH = Path("data/index_meta_enhanced.json")
TOPICS_JSON = Path("data/topics.json")

MODEL_NAME = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
GEN_OPTS = {"temperature": 0.0, "seed": 0, "top_k": 1, "top_p": 0.1}

# Enhanced negation and uncertainty patterns
NEG_TOKENS = {
    "no", "not", "never", "none", "without", "neither", "nor", "contraindicated", 
    "except", "lack", "absence", "unlikely", "rarely", "seldom", "exclude",
    "avoid", "prevent", "inhibit", "contra", "anti", "non", "un", "dis"
}

UNCERTAINTY_TOKENS = {
    "may", "might", "could", "possible", "possibly", "probably", "likely",
    "sometimes", "occasionally", "often", "usually", "typically", "generally",
    "appears", "seems", "suggests", "indicates", "potential", "risk", "chance"
}

QUANTIFIER_PATTERNS = {
    "universal": r"\b(all|every|always|never|none|100%)\b",
    "majority": r"\b(most|majority|typically|usually|>?\s*\d+%)\b", 
    "minority": r"\b(some|few|rarely|seldom|<?\s*\d+%)\b",
    "precise": r"\b\d+\.?\d*\s*(mg|g|kg|ml|l|mmHg|bpm|%|degrees?)\b"
}

MEDICAL_DOMAINS = {
    "cardiology": ["heart", "cardiac", "cardiovascular", "myocardial", "coronary", "arrhythmia", "ecg", "echo"],
    "pulmonology": ["lung", "pulmonary", "respiratory", "pneumonia", "copd", "asthma", "ventilation"],
    "neurology": ["brain", "neurological", "stroke", "seizure", "cva", "tia", "consciousness"],
    "emergency": ["trauma", "emergency", "acute", "critical", "shock", "resuscitation", "triage"],
    "pharmacology": ["drug", "medication", "dose", "therapy", "treatment", "administration", "contraindicated"]
}

# JSON schema for structured LLM output
SCHEMA = {
    "type": "object",
    "properties": {
        "statement_is_true": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": 200},
        "key_evidence": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}
    },
    "required": ["statement_is_true", "confidence"]
}

# ---------------- Load enhanced retrieval assets ----------------
print("Loading retrieval assets...")

with META_PATH.open() as fh:
    METADATA = json.load(fh)

# Extract chunks with enhanced metadata
CHUNKS = METADATA.get('chunks', {})
STATS = METADATA.get('stats', {})
TOPIC_MAPPING = METADATA.get('topic_mapping', {})

# Load topics JSON if separate
if TOPICS_JSON.exists() and not TOPIC_MAPPING:
    TOPIC_MAPPING = json.loads(TOPICS_JSON.read_text())

print(f"Loaded {len(CHUNKS)} chunks across {len(STATS.get('topics', {}))} topics")

# Initialize embedding model
EMBEDDER = SentenceTransformer(STATS.get('embedding_model', "intfloat/e5-large-v2"), device="cuda")
EMBED_DIM = EMBEDDER.get_sentence_embedding_dimension()

# Load HNSW index
HNSW = hnswlib.Index(space="cosine", dim=EMBED_DIM)
HNSW.load_index(str(INDEX_PATH))
HNSW.set_ef(128)  # High recall for feature extraction

print("✅ Retrieval assets loaded successfully")

# ---------------- Enhanced Helper Functions ----------------

def extract_medical_features(statement: str) -> Dict[str, float]:
    """Extract medical domain-specific linguistic features."""
    text_lower = statement.lower()
    features = {}
    
    # Domain classification
    domain_scores = {}
    for domain, keywords in MEDICAL_DOMAINS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        domain_scores[domain] = score
        features[f"domain_{domain}"] = float(score)
    
    # Primary domain
    primary_domain = max(domain_scores.keys(), key=lambda k: domain_scores[k])
    features["primary_domain_score"] = float(domain_scores[primary_domain])
    
    # Quantifier analysis
    for qtype, pattern in QUANTIFIER_PATTERNS.items():
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        features[f"quantifier_{qtype}"] = float(matches)
    
    # Uncertainty markers
    uncertainty_count = sum(1 for token in UNCERTAINTY_TOKENS 
                          if re.search(rf'\b{re.escape(token)}\b', text_lower))
    features["uncertainty_markers"] = float(uncertainty_count)
    
    # Negation complexity
    neg_positions = []
    words = statement.split()
    for i, word in enumerate(words):
        if word.lower().strip(".,!?;:()[]") in NEG_TOKENS:
            neg_positions.append(i / len(words))  # Normalized position
    
    features["negation_count"] = float(len(neg_positions))
    features["negation_early"] = float(any(pos < 0.3 for pos in neg_positions)) if neg_positions else 0.0
    features["negation_spread"] = float(np.std(neg_positions)) if len(neg_positions) > 1 else 0.0
    
    return features

def analyze_chunk_levels(retrieved_chunks: List[Dict]) -> Dict[str, float]:
    """Analyze distribution across hierarchical chunk levels."""
    if not retrieved_chunks:
        return {"level_context": 0.0, "level_standard": 0.0, "level_fact": 0.0, "level_diversity": 0.0}
    
    level_counts = Counter(chunk.get('level', 'standard') for chunk in retrieved_chunks)
    total = len(retrieved_chunks)
    
    features = {}
    for level in ['context', 'standard', 'fact']:
        features[f"level_{level}"] = float(level_counts.get(level, 0) / total)
    
    # Level diversity (entropy)
    if total > 0:
        probs = [count/total for count in level_counts.values()]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        features["level_diversity"] = entropy
    else:
        features["level_diversity"] = 0.0
    
    return features

def compute_semantic_coherence(statement: str, retrieved_texts: List[str]) -> Dict[str, float]:
    """Compute semantic coherence between statement and retrieved passages."""
    if not retrieved_texts:
        return {"semantic_coherence": 0.0, "max_coherence": 0.0, "coherence_std": 0.0}
    
    # Encode statement and passages
    statement_vec = EMBEDDER.encode(f"query: {statement}", normalize_embeddings=True)
    passage_vecs = EMBEDDER.encode([f"passage: {text}" for text in retrieved_texts], 
                                  normalize_embeddings=True)
    
    # Compute similarities
    similarities = np.dot(passage_vecs, statement_vec)
    
    return {
        "semantic_coherence": float(np.mean(similarities)),
        "max_coherence": float(np.max(similarities)),
        "coherence_std": float(np.std(similarities))
    }

def analyze_key_term_alignment(statement: str, retrieved_chunks: List[Dict]) -> Dict[str, float]:
    """Analyze alignment between statement and retrieved chunk key terms."""
    statement_lower = statement.lower()
    
    # Extract key terms from retrieved chunks
    all_key_terms = []
    for chunk in retrieved_chunks:
        all_key_terms.extend(chunk.get('key_terms', []))
    
    if not all_key_terms:
        return {"key_term_overlap": 0.0, "key_term_density": 0.0, "unique_key_terms": 0.0}
    
    # Count overlapping terms
    overlapping_terms = sum(1 for term in all_key_terms if term in statement_lower)
    unique_terms = len(set(all_key_terms))
    
    return {
        "key_term_overlap": float(overlapping_terms / len(all_key_terms)) if all_key_terms else 0.0,
        "key_term_density": float(overlapping_terms / len(statement.split())) if statement.split() else 0.0,
        "unique_key_terms": float(unique_terms)
    }

def create_enhanced_prompt(statement: str, retrieved_chunks: List[Dict]) -> List[Dict]:
    """Create enhanced prompt with hierarchical context and reasoning guidance."""
    
    # Organize chunks by level for structured presentation
    chunks_by_level = defaultdict(list)
    for chunk in retrieved_chunks:
        level = chunk.get('level', 'standard')
        chunks_by_level[level].append(chunk)
    
    # Build context sections
    context_sections = []
    
    # Add context-level chunks (broad understanding)
    if chunks_by_level['context']:
        context_text = "\n".join(f"Context {i+1}: {chunk['text'][:300]}..." 
                               for i, chunk in enumerate(chunks_by_level['context']))
        context_sections.append(f"## BROAD CONTEXT:\n{context_text}")
    
    # Add standard chunks (detailed information)
    if chunks_by_level['standard']:
        standard_text = "\n".join(f"Detail {i+1}: {chunk['text'][:200]}..." 
                                for i, chunk in enumerate(chunks_by_level['standard']))
        context_sections.append(f"## DETAILED INFORMATION:\n{standard_text}")
    
    # Add fact chunks (precise facts)
    if chunks_by_level['fact']:
        fact_text = "\n".join(f"Fact {i+1}: {chunk['text']}" 
                            for i, chunk in enumerate(chunks_by_level['fact']))
        context_sections.append(f"## SPECIFIC FACTS:\n{fact_text}")
    
    evidence = "\n\n".join(context_sections)
    
    system_prompt = """You are a medical fact-checking expert specializing in emergency healthcare.

TASK: Determine if the given medical statement is factually accurate based ONLY on the provided evidence.

ANALYSIS FRAMEWORK:
1. PRECISION: Check for exact matches in dosages, percentages, timeframes, and medical terminology
2. SCOPE: Verify if claims about "all patients" vs "some patients" vs "specific populations" are accurate
3. CONTEXT: Consider emergency vs non-emergency settings, adult vs pediatric populations
4. NEGATION: Pay careful attention to "not", "never", "contraindicated", "except" statements
5. QUANTIFIERS: Verify claims using "always", "usually", "rarely", "most", percentages

DECISION RULES:
- If evidence directly supports the statement: TRUE (1)
- If evidence contradicts the statement: FALSE (0)  
- If evidence is insufficient, ambiguous, or off-topic: FALSE (0)
- If statement makes universal claims but evidence shows exceptions: FALSE (0)
- If statement contains any factual inaccuracy: FALSE (0)

OUTPUT FORMAT: Provide confidence (0.0-1.0), brief reasoning, and key evidence indices that influenced your decision."""

    user_prompt = f"""MEDICAL STATEMENT TO VERIFY:
"{statement}"

MEDICAL EVIDENCE:
{evidence}

Analyze the statement against the evidence and provide your verdict with reasoning."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def parse_enhanced_response(response_text: str) -> Tuple[int, float, str, List[int]]:
    """Parse enhanced LLM response with error handling."""
    try:
        # Find JSON in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start != -1 and end > start:
            json_str = response_text[start:end]
            result = json.loads(json_str)
            
            verdict = int(result.get('statement_is_true', 0))
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', '')
            key_evidence = result.get('key_evidence', [])
            
            return verdict, confidence, reasoning, key_evidence
    except Exception as e:
        print(f"JSON parsing failed: {e}")
    
    # Fallback parsing
    verdict = 1 if 'true' in response_text.lower() else 0
    return verdict, 0.5, '', []

def entropy(counts: List[int]) -> float:
    """Compute entropy of count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    
    entropy_val = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy_val -= p * math.log(p + 1e-12)
    
    return entropy_val

# ---------------- Main Feature Computation ----------------

def compute_enhanced_features(statement: str, k: int = 8, use_hierarchical: bool = True) -> Tuple[Dict[str, float], List[int]]:
    """
    Compute comprehensive features leveraging enhanced indexing.
    
    Args:
        statement: Medical statement to analyze
        k: Number of passages to retrieve
        use_hierarchical: Whether to use hierarchical retrieval strategy
    
    Returns:
        Tuple of (features_dict, retrieved_passage_ids)
    """
    print(f"🔍 Computing features for statement: {statement[:100]}...")
    
    # 1) RETRIEVAL with hierarchical strategy
    query_vec = EMBEDDER.encode(f"query: {statement}", normalize_embeddings=True)
    labels, distances = HNSW.knn_query(np.asarray(query_vec, dtype="float32")[None, :], k=k*2)  # Retrieve more for filtering
    
    passage_ids = labels[0].tolist()
    distances = distances[0].tolist()
    similarities = [1.0 - d for d in distances]
    
    # Get retrieved chunks with metadata
    retrieved_chunks = []
    valid_ids = []
    valid_similarities = []
    
    for pid, sim in zip(passage_ids, similarities):
        chunk_data = CHUNKS.get(str(pid))
        if chunk_data:
            retrieved_chunks.append(chunk_data)
            valid_ids.append(pid)
            valid_similarities.append(sim)
    
    # Limit to k chunks
    retrieved_chunks = retrieved_chunks[:k]
    valid_ids = valid_ids[:k]
    valid_similarities = valid_similarities[:k]
    
    print(f"📖 Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # 2) LLM VERDICT with enhanced prompting
    try:
        messages = create_enhanced_prompt(statement, retrieved_chunks)
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=messages, 
            format=SCHEMA, 
            options=GEN_OPTS
        )
        
        response_content = response["message"]["content"]
        print(f"🤖 LLM Response: {response_content}")
        
        llm_verdict, confidence, reasoning, key_evidence = parse_enhanced_response(response_content)
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        llm_verdict, confidence, reasoning, key_evidence = 0, 0.5, '', []
    
    # 3) SIMILARITY FEATURES
    similarity_features = {}
    if valid_similarities:
        similarity_features.update({
            "sim_mean": float(np.mean(valid_similarities)),
            "sim_max": float(np.max(valid_similarities)),
            "sim_min": float(np.min(valid_similarities)),
            "sim_std": float(np.std(valid_similarities)),
            "sim_gap_top1_top2": float(valid_similarities[0] - valid_similarities[1]) if len(valid_similarities) >= 2 else 0.0,
            "sim_gap_top1_median": float(valid_similarities[0] - np.median(valid_similarities)),
            "sim_quartile_range": float(np.percentile(valid_similarities, 75) - np.percentile(valid_similarities, 25)) if len(valid_similarities) >= 4 else 0.0
        })
    else:
        similarity_features = {f"sim_{key}": 0.0 for key in ["mean", "max", "min", "std", "gap_top1_top2", "gap_top1_median", "quartile_range"]}
    
    # 4) TOPIC ANALYSIS
    topic_features = {}
    if retrieved_chunks:
        # Extract topics from chunks
        chunk_topics = []
        for chunk in retrieved_chunks:
            topic_name = chunk.get('topic_name', '')
            topic_id = TOPIC_MAPPING.get(topic_name, -1)
            if topic_id != -1:
                chunk_topics.append(topic_id)
        
        if chunk_topics:
            topic_counts = Counter(chunk_topics)
            topic_features.update({
                "vote_entropy": entropy(list(topic_counts.values())),
                "unique_topics": float(len(topic_counts)),
                "dominant_topic_ratio": float(max(topic_counts.values()) / len(chunk_topics)),
                "topic_agreement": float(max(topic_counts.values()) / len(chunk_topics))  # Same as dominant ratio
            })
        else:
            topic_features = {"vote_entropy": 0.0, "unique_topics": 0.0, "dominant_topic_ratio": 0.0, "topic_agreement": 0.0}
    else:
        topic_features = {"vote_entropy": 0.0, "unique_topics": 0.0, "dominant_topic_ratio": 0.0, "topic_agreement": 0.0}
    
    # 5) HIERARCHICAL CHUNK ANALYSIS
    level_features = analyze_chunk_levels(retrieved_chunks)
    
    # 6) MEDICAL LINGUISTIC FEATURES
    medical_features = extract_medical_features(statement)
    
    # 7) SEMANTIC COHERENCE
    retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    coherence_features = compute_semantic_coherence(statement, retrieved_texts)
    
    # 8) KEY TERM ALIGNMENT
    alignment_features = analyze_key_term_alignment(statement, retrieved_chunks)
    
    # 9) STATEMENT SURFACE FEATURES
    tokens = statement.lower().split()
    surface_features = {
        "stmt_len_words": float(len(tokens)),
        "stmt_len_chars": float(len(statement)),
        "avg_word_length": float(np.mean([len(word) for word in tokens])) if tokens else 0.0,
        "has_percent": float("%" in statement),
        "has_numbers": float(any(char.isdigit() for char in statement)),
        "num_digits": float(sum(char.isdigit() for char in statement)),
        "num_punctuation": float(sum(not char.isalnum() and not char.isspace() for char in statement)),
        "sentence_count": float(len(re.split(r'[.!?]+', statement.strip())))
    }
    
    # 10) LLM-DERIVED FEATURES
    llm_features = {
        "llm_verdict": float(llm_verdict),
        "llm_confidence": float(confidence),
        "has_reasoning": float(len(reasoning.strip()) > 0),
        "reasoning_length": float(len(reasoning.split())),
        "key_evidence_count": float(len(key_evidence))
    }
    
    # COMBINE ALL FEATURES
    all_features = {}
    all_features.update(similarity_features)
    all_features.update(topic_features)
    all_features.update(level_features)
    all_features.update(medical_features)
    all_features.update(coherence_features)
    all_features.update(alignment_features)
    all_features.update(surface_features)
    all_features.update(llm_features)
    
    print(f"✅ Computed {len(all_features)} features")
    
    return all_features, valid_ids

# ---------------- Batch Processing Support ----------------

def compute_features_batch(statements: List[str], k: int = 8) -> List[Tuple[Dict[str, float], List[int]]]:
    """Process multiple statements efficiently."""
    results = []
    
    print(f"🔄 Processing {len(statements)} statements in batch...")
    
    for i, statement in enumerate(statements):
        print(f"Processing {i+1}/{len(statements)}: {statement[:50]}...")
        try:
            features, pids = compute_enhanced_features(statement, k=k)
            results.append((features, pids))
        except Exception as e:
            print(f"❌ Error processing statement {i+1}: {e}")
            # Return empty features as fallback
            empty_features = {f: 0.0 for f in [
                "llm_verdict", "sim_mean", "sim_max", "sim_min", "sim_std",
                "vote_entropy", "unique_topics", "stmt_len_words", "stmt_len_chars"
            ]}
            results.append((empty_features, []))
    
    return results

# Legacy compatibility - maintain original function signature
def compute_features(statement: str, k: int = 4) -> Tuple[Dict[str, float], List[int]]:
    """Legacy compatibility wrapper."""
    return compute_enhanced_features(statement, k=k)