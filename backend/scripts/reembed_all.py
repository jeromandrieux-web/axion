# backend/scripts/reembed_all.py
# Ré-encode les "chunks" d'une affaire (backend/storage/cases/<case_id>/bm25.json) avec le modèle d'embeddings (E5),
# normalise les vecteurs et écrit faiss.index + vectors.npy dans le dossier de l'affaire.

import os, json, time, argparse
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")
PASSAGE_PREFIX = (os.getenv("E5_PASSAGE_PREFIX", "passage:") or "passage:").strip()
BATCH = int(os.getenv("REEMBED_BATCH", "64"))

def reembed_case(case_id: str, backup: bool = True):
    casedir = STORAGE / "cases" / case_id
    bm25_path = casedir / "bm25.json"
    if not bm25_path.exists():
        print(f"[SKIP] {case_id}: bm25.json manquant")
        return
    data = json.loads(bm25_path.read_text())
    chunks = data.get("chunks", [])
    if not chunks:
        print(f"[SKIP] {case_id}: 0 chunks")
        return

    # Préfixer chaque chunk en "passage:" pour e5
    texts = [
        (PASSAGE_PREFIX + " " + c) if not c.lower().startswith(PASSAGE_PREFIX) else c
        for c in chunks
    ]

    print(f"[{case_id}] Encodage {len(texts)} passages avec {EMB_MODEL_NAME} (batch={BATCH})…")
    t0 = time.time()
    model = SentenceTransformer(EMB_MODEL_NAME)
    X = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=BATCH,
    ).astype("float32")
    faiss.normalize_L2(X)

    # Sauvegardes si demandé
    faiss_path = casedir / "faiss.index"
    vecs_path  = casedir / "vectors.npy"
    if backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        if faiss_path.exists():
            faiss_path.rename(casedir / f"faiss.index.bak.{ts}")
        if vecs_path.exists():
            vecs_path.rename(casedir / f"vectors.npy.bak.{ts}")

    # Écriture index + vectors
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(faiss_path))
    np.save(vecs_path, X)

    dt = time.time() - t0
    print(f"[OK] {case_id}: {len(texts)} chunks ré-encodés en {dt:.1f}s")

def list_cases() -> list[str]:
    root = STORAGE / "cases"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def main():
    ap = argparse.ArgumentParser(description="Re-embed FAISS avec e5 (passage:)")
    ap.add_argument("--case", help="ID d'affaire à ré-encoder (sinon: toutes)")
    ap.add_argument("--no-backup", action="store_true", help="Ne pas faire de .bak")
    ap.add_argument("--batch", type=int, help="Taille de batch encode() (défaut via REEMBED_BATCH)")
    args = ap.parse_args()

    if args.batch:
        os.environ["REEMBED_BATCH"] = str(args.batch)

    if args.case:
        reembed_case(args.case, backup=not args.no_backup)
    else:
        for cid in list_cases():
            reembed_case(cid, backup=not args.no_backup)

if __name__ == "__main__":
    main()

