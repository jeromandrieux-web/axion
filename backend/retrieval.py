# backend/retrieval.py
# Moteur de recherche hybride (BM25 + dense) pour Axion
# Auteur: Jax
#
# ✅ Version améliorée (Axion) :
# - Correctif crucial : neighbor expand n’écrase plus la couverture (on ajoute "en plus")
# - MMR optimisé (évite normalisations répétées coûteuses / bug de diversité)
# - Dédoublonnage (seuil) appliqué au niveau chunk (et option au niveau unité)
# - Support des nouvelles métadonnées ingest: unit_id, unit_type, is_unit_anchor
# - Regroupement par unité : utilise unit_id si présent (sinon fallback (source,unit_index))
# - Modes "inventory/exhaustif" (intention) : +TOP_K, +L, -MAX_PER_DOC, -MMR diversité si besoin
#
# ⚠️ Rétro-compat : fonctionne même si bm25.json ancien (sans unit_id).

import os, re, json, unicodedata, logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from time import perf_counter

logger = logging.getLogger(__name__)

# ---------- Config depuis .env ----------
STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()

BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B  = float(os.getenv("BM25_B", "0.75"))

TOP_K_DEFAULT   = int(os.getenv("TOP_K", "12"))
RRF_K           = int(os.getenv("RRF_K", "60"))
MMR_LAMBDA      = float(os.getenv("MMR_LAMBDA", "0.7"))
NEIGHBOR_EXPAND = int(os.getenv("NEIGHBOR_EXPAND", "1"))
MAX_PER_DOC     = int(os.getenv("MAX_PER_DOC", "3"))
SEARCH_MULT     = int(os.getenv("SEARCH_MULT", "6"))

DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.97"))

# 🔁 IMPORTANT : on se cale sur un modèle d’embedding robuste et dispo
DEFAULT_EMB_MODEL = "intfloat/multilingual-e5-base"
EMB_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("EMB_MODEL", DEFAULT_EMB_MODEL),
)

# Pour e5, le préfixe recommandé est "query:" pour les requêtes
E5_QUERY_PREFIX = os.getenv("E5_QUERY_PREFIX", "query:").strip()

NAME_BOOST      = float(os.getenv("NAME_BOOST", "1.0"))
PHRASE_BOOST    = float(os.getenv("PHRASE_BOOST", "1.0"))

# ✅ Boost meta "identité/état-civil/signalement"
IDENTITY_BOOST  = float(os.getenv("IDENTITY_BOOST", "2.0"))

# ✅ Boost spécifique des chunks structurés (is_fact=True) générés par ingest.py
FACT_BOOST = float(os.getenv("FACT_BOOST", "4.0"))

# ✅ MMR dynamique pour requêtes identité (plus de pertinence, moins de diversité)
MMR_LAMBDA_IDENTITY = float(os.getenv("MMR_LAMBDA_IDENTITY", "0.78"))

# ✅ Mode "inventaire" : augmente la couverture (utile pour “liste tous les PV…”, “recense…”, etc.)
INVENTORY_TOPK_MULT = float(os.getenv("INVENTORY_TOPK_MULT", "3.0"))   # top_k *= ...
INVENTORY_L_MULT    = float(os.getenv("INVENTORY_L_MULT", "2.0"))      # L *= ...
INVENTORY_MAX_PER_DOC = int(os.getenv("INVENTORY_MAX_PER_DOC", "0"))   # 0 => pas de cap
INVENTORY_MMR_LAMBDA = float(os.getenv("INVENTORY_MMR_LAMBDA", "0.85"))# favorise pertinence

# ✅ On peut demander explicitement "group_by_unit" au retrieval (chat_service le fera)
GROUP_BY_UNIT_DEFAULT = os.getenv("GROUP_BY_UNIT_DEFAULT", "0") == "1"

IDENTITY_DOC_TYPES = {
    "pv_identite",
    "pv_identité",
    "identite",
    "identité",
    "etat_civil",
    "état_civil",
    "etat civil",
    "état civil",
    "signalement",
}

IDENTITY_QUERY_TERMS = [
    "identite", "identité", "etat civil", "état civil",
    "date de naissance", "naissance", "né le", "nee le", "née le",
    "lieu de naissance", "nationalite", "nationalité",
    "domicile", "adresse", "reside", "réside",
    "filiation", "pere", "père", "mere", "mère",
    "signalement", "taille", "poids", "cheveux", "yeux",
    "piece d identite", "pièce d identité", "carte nationale",
    "passeport", "cni", "nni",
]

INVENTORY_TERMS = [
    "liste", "lister", "recense", "recenser", "inventaire", "exhaustif", "exhaustive",
    "tous les", "toutes les", "l'ensemble", "la totalité", "global", "complet",
    "tous les pv", "toutes les pv", "tous les proces", "tous les procès",
    "tous les actes", "toutes les auditions", "toutes les perquisitions",
]

_EMB_MODEL: Optional[SentenceTransformer] = None
_BM25_CACHE: Dict[str, Any] = {}  # cache simple par case_id : {tokens, bm25_obj, chunks_len}


def _load_model() -> None:
    """Charge le modèle d'embedding avec fallback et cache global."""
    global _EMB_MODEL
    if _EMB_MODEL is not None:
        return

    # 1) tentative sur EMB_MODEL_NAME
    try:
        logger.info("Loading retrieval embedding model: %s", EMB_MODEL_NAME)
        _EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
        return
    except Exception:
        logger.exception("Failed to load retrieval embedding model '%s'.", EMB_MODEL_NAME)

    # 2) fallback sur le modèle par défaut si différent
    if EMB_MODEL_NAME != DEFAULT_EMB_MODEL:
        try:
            logger.info("Falling back to default retrieval embedding model: %s", DEFAULT_EMB_MODEL)
            _EMB_MODEL = SentenceTransformer(DEFAULT_EMB_MODEL)
            return
        except Exception:
            logger.exception("Failed to load default retrieval embedding model '%s'.", DEFAULT_EMB_MODEL)

    raise RuntimeError(
        f"Impossible de charger un modèle d'embedding pour retrieval. "
        f"EMBEDDING_MODEL/EMB_MODEL='{EMB_MODEL_NAME}', DEFAULT='{DEFAULT_EMB_MODEL}'."
    )


def _model_name_in_use() -> str:
    """Nom/chemin du modèle réellement chargé (important si fallback)."""
    global _EMB_MODEL
    if _EMB_MODEL is None:
        return ""
    return str(getattr(_EMB_MODEL, "model_name_or_path", "") or "")


def _is_e5_model_loaded() -> bool:
    """Détecte e5 sur le modèle réellement chargé."""
    name = (_model_name_in_use() or EMB_MODEL_NAME or "")
    return "e5" in name.lower()


def _prepare_query_for_embedding(query: str) -> str:
    """Préfixe 'query:' pour e5 si nécessaire."""
    if not query:
        return query or ""
    if _is_e5_model_loaded():
        pref_norm = norm_text(E5_QUERY_PREFIX or "")
        now_norm = norm_text(query)
        if pref_norm and now_norm.startswith(pref_norm):
            return query
        return (E5_QUERY_PREFIX + " " + query).strip()
    return query


# ---------- Normalisation / Tokenisation ----------
def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokenize_for_bm25(texts: List[str]) -> List[List[str]]:
    return [norm_text(t).split() for t in texts]


# ---------- RRF ----------
def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for rl in ranked_lists:
        for rank, doc_id in enumerate(rl, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _rank_by_name_boost(chunks: List[str], sources: List[Dict], query: str) -> List[int]:
    q = norm_text(query)
    toks = [t for t in re.split(r'[^a-z0-9_.-]+', q) if t]
    if not toks:
        return []
    scores: Dict[int, int] = {}
    for i, meta in enumerate(sources):
        name = norm_text(meta.get("source", ""))
        sc = sum(1 for t in toks if t and t in name)
        if sc > 0:
            scores[i] = sc
    if not scores:
        return []
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _extract_quoted_phrase(query: str) -> str:
    m = re.search(r'"([^"]{4,})"|\'([^\']{4,})\'', query)
    if m:
        return (m.group(1) or m.group(2) or "").strip()
    return ""


def _rank_by_phrase_match(chunks: List[str], phrase: str) -> List[int]:
    if not phrase:
        return []
    p = norm_text(phrase)
    hits: List[int] = []
    for i, c in enumerate(chunks):
        if p and p in norm_text(c):
            hits.append(i)
    return hits


# ---------- Identité ----------
def _query_looks_like_identity(query: str) -> bool:
    q = norm_text(query)
    if not q:
        return False

    if any(t in q for t in (norm_text(x) for x in IDENTITY_QUERY_TERMS)):
        return True

    if re.search(r"\b(ne|nee|née)\s+le\b", q):
        return True

    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", q):
        if any(k in q for k in ["naissance", "ne le", "nee le", "née le", "etat civil", "identite", "identité"]):
            return True

    if any(k in q for k in ["nom :", "prenom :", "prénom :", "filiation :", "adresse :"]):
        return True

    return False


def _query_looks_like_inventory(query: str) -> bool:
    q = norm_text(query)
    if not q:
        return False
    return any(norm_text(t) in q for t in INVENTORY_TERMS)


def _rank_by_identity_meta(sources: List[Dict], query: str) -> List[int]:
    if not _query_looks_like_identity(query):
        return []

    dt_set = {norm_text(x) for x in IDENTITY_DOC_TYPES}
    hits: List[Tuple[int, int]] = []

    for i, meta in enumerate(sources):
        dt = norm_text(str(meta.get("doc_type", "") or ""))
        ut = norm_text(str(meta.get("unit_title", "") or ""))

        score = 0
        if dt and dt in dt_set:
            score += 3

        if any(k in ut for k in ["identite", "identité", "etat civil", "état civil", "signalement", "filiation"]):
            score += 2

        srcname = norm_text(str(meta.get("source", "") or ""))
        if any(k in srcname for k in ["identite", "etat_civil", "etat civil", "signalement"]):
            score += 1

        if score > 0:
            hits.append((i, score))

    if not hits:
        return []

    hits.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in hits]


def _rank_by_fact_chunks(sources: List[Dict], query: str) -> List[int]:
    if FACT_BOOST <= 0:
        return []
    if not _query_looks_like_identity(query):
        return []

    hits: List[int] = []
    for i, meta in enumerate(sources):
        if meta.get("is_fact") is True:
            hits.append(i)
    return hits


# ---------- MMR diversité ----------
def mmr_select(
    candidates: List[int],
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    top_k: int,
    lamb: float = 0.7
) -> List[int]:
    """
    MMR : maximise pertinence vs diversité.
    Optimisé : calcule une fois les similarités chunk-chunk.
    """
    if not candidates:
        return []

    qv = query_vec.astype("float32", copy=False)
    faiss.normalize_L2(qv)

    X = embeddings[candidates].astype("float32", copy=False)
    faiss.normalize_L2(X)

    sim_q = (X @ qv.T).flatten()

    selected: List[int] = []
    selected_local: List[int] = []  # indices locaux (dans candidates)

    # precompute sims entre candidats (peut être lourd si huge L ; L est déjà limité)
    sims = X @ X.T  # (Lcand, Lcand)

    for _ in range(min(top_k, len(candidates))):
        best_j = None
        best_score = -1e9

        for j in range(len(candidates)):
            if j in selected_local:
                continue

            if not selected_local:
                div_pen = 0.0
            else:
                div_pen = float(sims[j, selected_local].max())

            score = lamb * float(sim_q[j]) - (1.0 - lamb) * div_pen
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            break
        selected_local.append(best_j)
        selected.append(candidates[best_j])

    return selected


# ---------- Limites / Répartition par doc ----------
def _enforce_max_per_doc(ids: List[int], sources: List[Dict], max_per: int) -> List[int]:
    if max_per <= 0:
        return ids
    out: List[int] = []
    per: Dict[str, int] = {}
    for i in ids:
        src = sources[i].get("source", "")
        if per.get(src, 0) >= max_per:
            continue
        out.append(i)
        per[src] = per.get(src, 0) + 1
    return out


def _round_robin_by_doc(ids: List[int], sources: List[Dict]) -> List[int]:
    from collections import defaultdict, deque
    buckets = defaultdict(deque)
    for i in ids:
        buckets[sources[i].get("source", "")].append(i)
    keys = list(buckets.keys())
    pos = 0
    out: List[int] = []
    while any(buckets[k] for k in keys):
        k = keys[pos % len(keys)]
        if buckets[k]:
            out.append(buckets[k].popleft())
        pos += 1
    return out


# ---------- Dédoublonnage (texte) ----------
def _dedup_by_text(ids: List[int], chunks: List[str], threshold: float = DEDUP_THRESHOLD) -> List[int]:
    """
    Dédoublonne naïvement sur la normalisation + hash.
    (Seuil conservé pour compat, mais ici on utilise surtout une approche stable)
    """
    if not ids:
        return []
    seen = set()
    out: List[int] = []
    for i in ids:
        key = norm_text(chunks[i])
        if not key:
            continue
        h = hash(key)
        if h in seen:
            continue
        seen.add(h)
        out.append(i)
    return out


# ---------- Neighbor expand (FIX couverture) ----------
def _with_neighbors_plus(
    base_ids: List[int],
    chunks: List[str],
    sources: List[Dict],
    *,
    top_k: int,
    extra_neighbors: Optional[int] = None,
    max_per_doc: Optional[int] = None,
) -> List[Tuple[str, Dict]]:
    """
    Fix majeur : au lieu de remplir jusqu’à top_k en mélangeant voisins et base,
    on :
    1) garde la liste base_ids (coverage)
    2) ajoute les voisins autour (context local)
    3) renvoie jusqu’à (top_k + extra_neighbors) pour alimenter le prompt RAG

    extra_neighbors : nombre maximum de chunks additionnels au-delà de top_k.
    Si None : on prend top_k (donc jusqu’à 2*top_k au total), raisonnable.
    """
    if not base_ids:
        return []

    extra = int(extra_neighbors) if extra_neighbors is not None else int(top_k)
    limit_total = int(top_k) + max(0, extra)

    picked = set()
    per_doc: Dict[str, int] = {}
    results: List[Tuple[str, Dict]] = []

    cap = MAX_PER_DOC if max_per_doc is None else int(max_per_doc)

    def can_take(j: int) -> bool:
        if j in picked:
            return False
        src = sources[j].get("source", "")
        if cap > 0 and per_doc.get(src, 0) >= cap:
            return False
        return True

    def take(j: int) -> None:
        picked.add(j)
        results.append((chunks[j], sources[j]))
        src = sources[j].get("source", "")
        per_doc[src] = per_doc.get(src, 0) + 1

    # 1) d’abord les base_ids (couverture)
    for idx in base_ids:
        if 0 <= idx < len(chunks) and can_take(idx):
            take(idx)
            if len(results) >= limit_total:
                return results

    # 2) ensuite les voisins (contexte local)
    for idx in base_ids:
        for off in range(-NEIGHBOR_EXPAND, NEIGHBOR_EXPAND + 1):
            if off == 0:
                continue
            j = idx + off
            if 0 <= j < len(chunks) and can_take(j):
                take(j)
                if len(results) >= limit_total:
                    return results

    return results


# ---------- Récupération hybride ----------
def retrieve_hybrid(
    case_id: str,
    query: str,
    top_k: int = TOP_K_DEFAULT,
    **kwargs: Any,
) -> List[Any]:
    """
    Récupération hybride (BM25 + dense + boosts).
    - Mode historique : List[Tuple[str, Dict]]
    - Mode chat : List[Dict[str, Any]]
    """
    max_doc_chars: Optional[int] = kwargs.get("max_doc_chars")
    max_rag_chars: Optional[int] = kwargs.get("max_rag_chars")
    user_message_history = kwargs.get("user_message_history")  # accepté mais non utilisé

    # nouveaux flags
    group_by_unit: bool = bool(kwargs.get("group_by_unit", GROUP_BY_UNIT_DEFAULT))
    inventory_mode: Optional[bool] = kwargs.get("inventory_mode")
    extra_neighbors: Optional[int] = kwargs.get("extra_neighbors")

    casedir = STORAGE / "cases" / case_id
    bm25_path = casedir / "bm25.json"
    if not bm25_path.exists():
        raise FileNotFoundError(f"bm25.json absent pour l'affaire {case_id} ({bm25_path})")

    data = json.loads(bm25_path.read_text(encoding="utf-8"))
    chunks: List[str] = data.get("chunks", [])
    sources: List[Dict] = data.get("sources", [])

    if not chunks:
        return []

    is_identity_query = _query_looks_like_identity(query)

    # auto inventory si non précisé
    if inventory_mode is None:
        inventory_mode = _query_looks_like_inventory(query)

    # paramètres dynamiques couverture
    dyn_top_k = int(max(1, top_k))
    dyn_L = int(max(SEARCH_MULT * dyn_top_k, dyn_top_k))
    dyn_max_per_doc = MAX_PER_DOC
    dyn_mmr_lambda = MMR_LAMBDA_IDENTITY if is_identity_query else MMR_LAMBDA

    if inventory_mode:
        dyn_top_k = int(max(1, round(dyn_top_k * INVENTORY_TOPK_MULT)))
        dyn_L = int(max(dyn_L, round(dyn_L * INVENTORY_L_MULT)))
        dyn_max_per_doc = INVENTORY_MAX_PER_DOC
        dyn_mmr_lambda = INVENTORY_MMR_LAMBDA

    # --- BM25 (avec cache tokens + bm25) ---
    t0 = perf_counter()

    cache = _BM25_CACHE.get(case_id)
    toks = None
    bm25 = None
    if cache and cache.get("chunks_len") == len(chunks):
        toks = cache.get("tokens")
        bm25 = cache.get("bm25")
    if toks is None:
        toks = tokenize_for_bm25(chunks)
        bm25 = BM25Okapi(toks, k1=BM25_K1, b=BM25_B)
        _BM25_CACHE[case_id] = {"tokens": toks, "bm25": bm25, "chunks_len": len(chunks)}

    qn = norm_text(query)
    bm_scores = bm25.get_scores(qn.split())
    bm_order = list(np.argsort(-bm_scores))[:dyn_L]
    t1 = perf_counter()

    # --- Dense (FAISS) ---
    dense_order: List[int] = []
    try:
        index_path = casedir / "faiss.index"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            _load_model()
            qtxt = _prepare_query_for_embedding(query)

            qvec = np.array(
                _EMB_MODEL.encode([qtxt], convert_to_numpy=True, normalize_embeddings=True)
            ).astype("float32")
            faiss.normalize_L2(qvec)

            if getattr(index, "d", None) != qvec.shape[1]:
                logger.warning(
                    "FAISS dim mismatch: index.d=%s vs qvec=%s (case=%s). Dense skipped.",
                    getattr(index, "d", None),
                    qvec.shape[1],
                    case_id,
                )
                dense_order = []
            else:
                _, I = index.search(qvec, dyn_L)
                dense_order = [int(i) for i in I[0] if int(i) >= 0]
        else:
            dense_order = []
    except Exception:
        logger.exception("Erreur lors de la recherche dense FAISS pour l'affaire %s", case_id)
        dense_order = []
    t2 = perf_counter()

    # --- Bonus: nom de fichier mentionné ---
    name_hits = _rank_by_name_boost(chunks, sources, query)[:dyn_L]

    # --- Bonus: phrase exacte entre guillemets ---
    phrase = _extract_quoted_phrase(query)
    phrase_hits = _rank_by_phrase_match(chunks, phrase)[:dyn_L] if phrase else []

    # --- Bonus: identité (métadonnées) ---
    identity_hits = _rank_by_identity_meta(sources, query)[:dyn_L]

    # --- Bonus: facts chunks (is_fact=True) ---
    fact_hits = _rank_by_fact_chunks(sources, query)[:dyn_L]

    # --- Fusion RRF ---
    lists: List[List[int]] = [bm_order]
    if dense_order:
        lists.append(dense_order)

    if name_hits:
        lists.append(name_hits * int(max(1, round(NAME_BOOST))))
    if phrase_hits:
        lists.append(phrase_hits * int(max(1, round(PHRASE_BOOST))))
    if identity_hits and IDENTITY_BOOST > 0:
        lists.append(identity_hits * int(max(1, round(IDENTITY_BOOST))))
    if fact_hits and FACT_BOOST > 0:
        lists.append(fact_hits * int(max(1, round(FACT_BOOST))))

    fused_scores = reciprocal_rank_fusion(lists, k=RRF_K)
    fused_sorted = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda kv: -kv[1])]
    cand_ids = fused_sorted[:dyn_L]

    # dédup basique (évite que l’ancre + la page identique se répètent trop)
    cand_ids = _dedup_by_text(cand_ids, chunks, threshold=DEDUP_THRESHOLD)

    # --- MMR ---
    vec_path = casedir / "vectors.npy"
    if vec_path.exists():
        vecs = np.load(vec_path).astype("float32")
        try:
            _load_model()
            qtxt = _prepare_query_for_embedding(query)
            qvec = np.array(
                _EMB_MODEL.encode([qtxt], convert_to_numpy=True, normalize_embeddings=True)
            ).astype("float32")
            faiss.normalize_L2(qvec)

            if vecs.ndim != 2 or vecs.shape[1] != qvec.shape[1]:
                logger.warning(
                    "vectors.npy dim mismatch: vecs=%s, qvec=%s (case=%s). MMR skipped.",
                    vecs.shape,
                    qvec.shape,
                    case_id,
                )
                mmr_ids = cand_ids[:dyn_top_k]
            else:
                mmr_ids = mmr_select(cand_ids, vecs, qvec, dyn_top_k, lamb=dyn_mmr_lambda)
        except Exception:
            logger.exception("Erreur lors du MMR, on retombe sur les candidats bruts.")
            mmr_ids = cand_ids[:dyn_top_k]
    else:
        mmr_ids = cand_ids[:dyn_top_k]

    # --- Round-robin + cap MAX_PER_DOC (dynamique selon inventory) ---
    rr_ids = _round_robin_by_doc(mmr_ids, sources)
    capped = _enforce_max_per_doc(rr_ids, sources, dyn_max_per_doc)

    # --- Neighbor expand (FIX) : ajoute au-delà de top_k au lieu d’écraser ---
    results: List[Tuple[str, Dict]] = _with_neighbors_plus(
        capped,
        chunks,
        sources,
        top_k=dyn_top_k,
        extra_neighbors=extra_neighbors,
        max_per_doc=dyn_max_per_doc,
    )

    t3 = perf_counter()
    logger.debug(
        "retrieve_hybrid(%s): BM25=%.3fs, dense=%.3fs, total=%.3fs, top_k=%d inv=%s",
        case_id,
        t1 - t0,
        t2 - t1,
        t3 - t0,
        dyn_top_k,
        bool(inventory_mode),
    )

    # --- Mode historique ---
    if max_doc_chars is None and max_rag_chars is None and not user_message_history and not group_by_unit:
        return results

    # --- Mode group_by_unit (pour chat_service ou UI) ---
    if group_by_unit:
        return group_results_by_unit(results)

    # --- Mode chat (liste de docs) ---
    docs: List[Dict[str, Any]] = []
    total_chars = 0

    # NB : results peut contenir > top_k (neighbors en plus). On respecte max_rag_chars.
    for text, meta in results:
        t = text or ""

        if max_doc_chars is not None and max_doc_chars > 0 and len(t) > max_doc_chars:
            t = t[:max_doc_chars]

        if max_rag_chars is not None:
            if total_chars >= max_rag_chars:
                break
            remaining = max_rag_chars - total_chars
            if len(t) > remaining:
                t = t[:remaining]
            total_chars += len(t)

        docs.append({
            "text": t,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id"),
            "local_chunk_id": meta.get("local_chunk_id"),
            "meta": meta,
        })

        if max_rag_chars is not None and total_chars >= max_rag_chars:
            break

    return docs


# ---------- Regroupement par unité logique ----------
def group_results_by_unit(results: List[Tuple[str, Dict]]) -> List[Dict[str, Any]]:
    """
    Regroupe les résultats (texte, meta) par unité logique.
    Priorité : unit_id si présent, sinon fallback (source, unit_index).
    """
    if not results:
        return []

    groups: Dict[Any, Dict[str, Any]] = {}

    for rank, (text, meta) in enumerate(results):
        src = meta.get("source", "")
        unit_id = meta.get("unit_id")  # nouveau ingest.py
        unit_idx = meta.get("unit_index", -1)

        key = unit_id if unit_id else (src, unit_idx)

        if key not in groups:
            groups[key] = {
                "source": src,
                "unit_id": unit_id,
                "unit_index": unit_idx,
                "unit_title": meta.get("unit_title"),
                "unit_type": meta.get("unit_type"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "date": meta.get("date"),
                "doc_type": meta.get("doc_type"),
                "rank": rank,
                "chunks": [],
            }

        groups[key]["chunks"].append({
            "text": text,
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id"),
            "local_chunk_id": meta.get("local_chunk_id"),
            "is_unit_anchor": bool(meta.get("is_unit_anchor")),
            "is_fact": bool(meta.get("is_fact")),
            "meta": meta,
        })

    return sorted(groups.values(), key=lambda g: g.get("rank", 0))
