#!/usr/bin/env python3
"""
One–off utility:
    python build_index.py \
        --docs_root data/topics \
        --chunk_size 256 \
        --embed_model intfloat/e5-large-v2 \
        --index_out data/index.hnsw \
        --meta_out  data/index_meta.json

Builds an HNSW vector index (+ JSON metadata) from the StatPearls markdown
corpus.  Takes ≈5 min on a typical laptop CPU for ~60 k passages.
"""
import argparse, json, os, re, pathlib
from typing import List, Dict, Tuple
from tqdm import tqdm

import hnswlib
from sentence_transformers import SentenceTransformer

# ----------------------------------------
SECTION_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
FRONT_MATTER_RE = re.compile(r"^---.*?^---\s*", re.DOTALL | re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

def clean_markdown(md: str) -> str:
    md = FRONT_MATTER_RE.sub("", md)
    md = CODE_BLOCK_RE.sub("", md)
    md = SECTION_RE.sub("", md)
    return re.sub(r"\s{2,}", " ", md).strip()

def chunk_text(txt: str, max_tokens: int, overlap: int = 50) -> List[str]:
    words = txt.split()
    stride = max_tokens - overlap
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), stride)
        if i + overlap < len(words)
    ] or [txt]

def walk_markdown_files(root: str) -> List[Tuple[str, str]]:
    md_files = []
    for topic in sorted(os.listdir(root)):
        tp = pathlib.Path(root) / topic
        if not tp.is_dir(): continue
        for md in tp.glob("*.md"):
            md_files.append((topic, str(md)))
    return md_files

def main(args: argparse.Namespace):
    print("🔍  Collecting markdown files …")
    docs = walk_markdown_files(args.docs_root)

    passages: List[str] = []
    meta: Dict[int, Dict] = {}
    pid = 0

    print("✂️   Cleaning & chunking …")
    for topic_name, path in tqdm(docs):
        with open(path, encoding="utf-8") as fh:
            txt = clean_markdown(fh.read())
        for chunk in chunk_text(txt, args.chunk_size, overlap=args.overlap):  # <<< use CLI overlap
            passages.append(chunk)
            meta[pid] = {"text": chunk, "topic_name": topic_name, "source": path}
            pid += 1

    print(f"📝  {len(passages):,} passages")

    print(f"📐  Embedding with {args.embed_model} …")
    embedder = SentenceTransformer(args.embed_model, device="cuda")  # <<< use "cuda"
    enc_inputs = [f"passage: {p}" for p in passages]                 # keep the e5 'passage:' prefix
    vecs = embedder.encode(
        enc_inputs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = vecs.shape[1]
    print(f"🏗️   Building HNSW index (dim={dim}) …")
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=vecs.shape[0], ef_construction=400, M=64)
    index.add_items(vecs)
    index.set_ef(50)
    index.save_index(args.index_out)
    print(f"💾  Saved HNSW index ➜ {args.index_out}")

    with open(args.meta_out, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"💾  Saved metadata ➜ {args.meta_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs_root", required=True)
    p.add_argument("--chunk_size", type=int, default=320)
    p.add_argument("--overlap", type=int, default=64)      # <<< NEW
    p.add_argument("--embed_model", default="mghuibregtse/biolinkbert-large-simcse-rat")
    p.add_argument("--index_out", required=True, help="e.g. data/index.hnsw")
    p.add_argument("--meta_out", required=True)
    main(p.parse_args())