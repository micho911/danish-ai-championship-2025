# Danish Championship in AI 2025 (DM i AI)

Solo entry in **[DM i AI 2025](https://dmiai.dk)**, the Danish national AI championship. I worked on three applied ML problems in different fields: **medical NLP with RAG and LLMs**, **reinforcement learning for autonomous driving**, and **medical image segmentation**. Training ran on **GPU clusters (HPC)**, and live inference APIs answered the competition's evaluation servers.

> The event was designed for teams of 4–6, and I competed alone. None of the three solutions is fully finished. This repository shows the methods, engineering and infrastructure I built for problem types I had never worked on before.

---

## Skills and technologies

| Area | What I used | Where |
|---|---|---|
| **Language and tooling** | Python, NumPy, pandas, SciPy, Jupyter, argparse CLIs, logging | everywhere |
| **Deep learning** | PyTorch, PyTorch Lightning, mixed-precision training, cosine LR scheduling, early stopping and checkpointing | [tumor-segmentation](tumor-segmentation/), [race-car](race-car/trainer.py) |
| **LLMs and NLP** | Hugging Face Transformers, PubMedBERT fine-tuning, **LoRA / PEFT**, multi-task classification heads, local LLM inference with **Ollama** (DeepSeek-R1 32B) | [emergency-healthcare-rag](emergency-healthcare-rag/) |
| **RAG and vector search** | Document chunking, dense embeddings (Sentence-Transformers, `e5-large-v2`), **HNSW** approximate nearest-neighbour index (`hnswlib`), retrieval-based topic voting | [build_index.py](emergency-healthcare-rag/build_index.py), [rag_model2.py](emergency-healthcare-rag/rag_model2.py) |
| **Classical ML** | scikit-learn: logistic regression, gradient boosting, voting ensembles, probability calibration, feature selection (SelectKBest, RFE), stratified k-fold CV, hyperparameter grid search, feature engineering | [train_truth_meta2.py](emergency-healthcare-rag/train_truth_meta2.py) |
| **Reinforcement learning** | **PPO** (Stable-Baselines3), custom **Gymnasium** environment, reward shaping, observation normalisation (VecNormalize), parallel environments (SubprocVecEnv), safety shielding | [race-car/evaluate.py](race-car/evaluate.py), [race-car/api.py](race-car/api.py) |
| **Evolutionary methods and control** | Neuroevolution (genetic algorithm over neural policies) with multi-GPU parallel rollouts, rule-based planner tuned by grid search | [trainer.py](race-car/trainer.py), [lane_planner.py](race-car/lane_planner.py) |
| **Computer vision and medical imaging** | U-Net with attention gates, pretrained EfficientNet/ResNet encoders (`segmentation_models_pytorch`), Dice + BCE loss for class imbalance, Albumentations augmentation, OpenCV, SimpleITK, **nnU-Net v2** dataset preparation | [tumor-segmentation](tumor-segmentation/) |
| **HPC and GPU compute** | [UCloud](https://cloud.sdu.dk) HPC, **NVIDIA (CUDA)** and **AMD (ROCm)** GPUs, **SLURM** batch jobs, multi-GPU process parallelism, headless simulation | [train_unet.slurm](tumor-segmentation/slurm/train_unet.slurm) |
| **Model serving and deployment** | **FastAPI** + Uvicorn REST endpoints, Pydantic schemas, **nginx reverse proxy** routing competition traffic from dmiai.dk to the model APIs, latency-constrained inference | `*/api.py`, [deployment/nginx.conf](deployment/nginx.conf) |

---

## The three challenges

### 1. Emergency Healthcare RAG ([details](emergency-healthcare-rag/))
*Given a medical statement, classify it as true or false and assign it to one of 115 emergency-medicine topics, using only locally hosted models.*

- Built a **retrieval-augmented pipeline** over the StatPearls medical corpus: chunking, then `e5-large-v2` embeddings, then an HNSW vector index.
- Predicted topics by majority vote over the retrieved passages, and truth with a **meta-classifier** trained on engineered features: retrieval similarity, negation and uncertainty cues, and a verdict from a **local LLM served through Ollama**.
- First attempt: **fine-tuned PubMedBERT** with LoRA and two classification heads, before moving to the retrieval approach because there were only 200 labelled examples.

### 2. Race Car ([details](race-car/))
*Drive a car through traffic as far as possible using only 16 distance sensors.*

- Wrapped the game in a **Gymnasium environment** and trained a **PPO** agent with Stable-Baselines3 across parallel environments.
- Served the policy behind a **safety shield** that caps speed, blocks unsafe acceleration and steers away from walls.
- Compared it against a **rule-based lane planner** tuned by grid search and a **genetic algorithm** evolving neural policies across multiple GPUs.

### 3. Tumor Segmentation ([details](tumor-segmentation/))
*Segment tumours in whole-body PET MIP scans, scored by Dice coefficient.*

- Trained an **attention U-Net with a pretrained EfficientNet encoder** in PyTorch Lightning, using Dice + BCE loss, mixed precision, augmentation and healthy controls as negative examples.
- Wrote **SLURM** jobs for GPU training on HPC, a hybrid ResNet U-Net pipeline, and a converter to the **nnU-Net v2** dataset format.

---

## Infrastructure

```
dmiai.dk evaluation server
          │  POST /predict
          ▼
   nginx reverse proxy (UCloud)      ← deployment/nginx.conf
          │
          ▼
   FastAPI model service ── GPU (NVIDIA / AMD) on UCloud HPC
```

Models were trained and served on UCloud GPU nodes. Each challenge exposed a FastAPI `/predict` endpoint, and nginx sat in front of it to handle large base64 image payloads, timeouts and routing for the competition's live evaluation.

---

## Repository layout

```
emergency-healthcare-rag/    RAG pipeline, meta-classifier, v1-pubmedbert-finetune/
race-car/                    PPO + Gymnasium env, lane planner, genetic trainer
tumor-segmentation/          Lightning U-Net, nnU-Net prep, SLURM job
deployment/nginx.conf        Reverse proxy config
```

Each challenge folder has a `README.md` describing my approach and a `TASK.md` with the official problem statement.

## Notes

- **Work in progress, left as it was.** Some scripts still reference module names from mid-iteration (for example `enhanced_*` vs. `*2.py` in the RAG folder). I haven't fixed them after the fact, so the repo shows where things stood when the competition ended.
- **No data or weights.** The datasets (patient PET scans, StatPearls articles) belong to the organizers and are available from [amboltio/DM-i-AI-2025](https://github.com/amboltio/DM-i-AI-2025).
- **Starter code.** The API scaffolding (`api.py`, `dtos.py`, `utils.py`, `example.py`), the race-car game engine (`race-car/src/`) and the `TASK.md` files come from the organizers' template. All modelling, training, retrieval, RL and deployment code is my own.
