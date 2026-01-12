# backend/chat_service.py
# Service de chat avec gestion du contexte, historique, RAG, et streaming.
# Auteur: Jax
#
# ✅ Version améliorée (Axion) :
# - Utilise retrieve_hybrid(group_by_unit=True) pour raisonner “par acte / PV”
# - Construit un contexte RAG balisé et citable (format [source:..., chunk ...] conservé)
# - Active automatiquement le mode inventaire pour les requêtes exhaustives
# - Augmente intelligemment le budget RAG (sans casser tes variables .env)
# - Améliore l’index des sources (inclut unit_id/unit_type/pages, is_unit_anchor/is_fact)
# - Streaming : citations détectées même si l’assistant cite “chunk_id” provenant d’un groupe
#
# ⚠️ Rétro-compat : si retrieval renvoie l’ancien format (docs dict), ça marche aussi.

import os
import re
import logging
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple, Any, Union

from fastapi.responses import StreamingResponse

from prompts import build_prompt
from extraction_prompts import FORMAT_INSTRUCTIONS_EXTRACT
from retrieval import retrieve_hybrid
from ollama_client import generate, stream_generate
from llm_profiles import get_profile
from context_utils import sse
from identity_mapper import anonymize_prompt_for_model, deanonymize_text_for_case

logger = logging.getLogger("axion.chat")

# Budgets de contexte configurables
MAX_DOC_CONTEXT_CHARS = int(os.getenv("CHAT_DOC_MAX_CHARS", "80000"))
MAX_RAG_CONTEXT_CHARS = int(os.getenv("CHAT_RAG_MAX_CHARS", "60000"))

# ✅ Ajouts : budget "inventaire" (si l'utilisateur demande exhaustif)
MAX_RAG_CONTEXT_CHARS_INVENTORY = int(os.getenv("CHAT_RAG_MAX_CHARS_INVENTORY", str(MAX_RAG_CONTEXT_CHARS * 3)))
MAX_DOC_CONTEXT_CHARS_INVENTORY = int(os.getenv("CHAT_DOC_MAX_CHARS_INVENTORY", str(MAX_DOC_CONTEXT_CHARS * 2)))

# ✅ Ajout : nombre max de chunks par unité à injecter (pour garder de la diversité)
MAX_CHUNKS_PER_UNIT_IN_CONTEXT = int(os.getenv("CHAT_MAX_CHUNKS_PER_UNIT", "6"))

# ✅ Ajout : inclure systématiquement les ancres d’actes si présentes
PREFER_UNIT_ANCHORS = os.getenv("CHAT_PREFER_UNIT_ANCHORS", "1") == "1"


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _query_looks_like_inventory(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    terms = [
        "liste", "lister", "recense", "recenser", "inventaire", "exhaustif", "exhaustive",
        "tous les", "toutes les", "l'ensemble", "la totalité", "global", "complet",
        "tous les pv", "toutes les pv", "tous les procès", "toutes les auditions",
        "toutes les perquisitions", "tous les actes", "toutes les actes",
    ]
    return any(t in qn for t in terms)


class ChatService:
    def __init__(self):
        # pattern pour détecter des références de type [source: xxx, chunk 12]
        self.cite_pattern = re.compile(
            r"\[source:\s*([^,\]]+)\s*,\s*chunk\s*(\d+)\s*\]",
            re.IGNORECASE,
        )

    # ---------- helpers internes ----------

    @staticmethod
    def _approx_tokens(s: str) -> int:
        """Estimation grossière #tokens."""
        if not s:
            return 0
        if not isinstance(s, str):
            s = str(s)
        return max(1, len(s) // 4)

    def _build_history_block(self, history: Optional[List[Any]]) -> str:
        """
        Construit un bloc texte à partir de l'historique.
        Limité par HISTORY_BUDGET_TOKENS, et tronque le dernier message si nécessaire.
        """
        if not history:
            return ""

        try:
            budget = int(os.getenv("HISTORY_BUDGET_TOKENS", "4096"))
        except Exception:
            budget = 4096

        parts: List[str] = []
        used = 0

        # on part des messages les plus récents
        for m in reversed(history):
            if hasattr(m, "role"):
                role = m.role
                content = m.content
            elif isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
            else:
                continue

            content = (content or "").strip()
            if not content:
                continue

            if role == "user":
                prefix = "Utilisateur : "
            elif role == "assistant":
                prefix = "Assistant : "
            else:
                prefix = "Système : "

            chunk = f"{prefix}{content}\n"
            need = self._approx_tokens(chunk)

            if used + need > budget:
                # premier message déjà trop gros → on tronque
                if used == 0:
                    avail = max(budget - self._approx_tokens(prefix), 16)
                    approx_chars = avail * 4
                    truncated = content[-approx_chars:]
                    chunk = f"{prefix}{truncated}\n"
                    need = self._approx_tokens(chunk)
                    if need > 0:
                        parts.append(chunk)
                        used += need
                break

            parts.append(chunk)
            used += need

        if not parts:
            return ""

        parts.reverse()

        return (
            "Tu disposes ci-dessous de l'historique récent de la conversation.\n"
            "Tu DOIS t'y référer pour comprendre le contexte, corriger ou détailler tes réponses précédentes,\n"
            "et adapter ta nouvelle réponse aux demandes de l'utilisateur.\n"
            "Ne dis jamais qu'il n'y a pas d'historique ou que tu ne disposes d'aucun contexte :\n"
            "même s'il est bref, l'historique fourni ci-dessous existe bel et bien.\n\n"
            "Historique de la conversation :\n"
            + "".join(parts).strip()
            + "\n\n"
        )

    @contextmanager
    def _temp_env(self, overrides: Dict[str, str]):
        """
        Utilitaire pour surcharger temporairement des variables d'environnement
        (utile pour ajuster les paramètres de retrieval).
        """
        old: Dict[str, Optional[str]] = {}
        try:
            for k, v in overrides.items():
                old[k] = os.environ.get(k)
                os.environ[k] = str(v)
            yield
        finally:
            for k, prev in old.items():
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev

    # ---------- logique principale ----------

    async def chat(self, req) -> dict:
        """Requête de chat non-streamée."""
        history = getattr(req, "history", None) or []
        user_msg = getattr(req, "message", "") or ""
        use_memory = self._use_memory(req)

        # 1) Préparer le contexte "brut" (doc_text ou RAG)
        ctx_raw, sources = self._prepare_context(req)

        # 2) Construire le contexte final en fonction du mode
        if use_memory:
            ctx_for_prompt = self._build_context_with_history(
                ctx_raw=ctx_raw,
                history=history,
                user_msg=user_msg,
                use_memory=True,
            )
        else:
            ctx_for_prompt = ctx_raw

        mode = getattr(req, "mode", None) or "plain"

        # 🔥 Mode "libre" : pas de mémoire RAG (use_memory=False) et pas de doc/RAG (ctx_raw vide)
        open_mode = (not use_memory) and not (ctx_raw or "").strip()

        if open_mode:
            final_prompt = self._build_open_prompt(user_msg=user_msg, history=history)
        else:
            # Mode avec contexte (affaire/RAG ou doc ponctuel)
            base_prompt = build_prompt(
                ctx_for_prompt,
                user_msg,
                lite=(not use_memory),
            )
            final_prompt = base_prompt

            if getattr(req, "system", None):
                final_prompt = req.system.strip() + "\n\n" + final_prompt

            if mode in ("extract", "extraction"):
                final_prompt = f"{final_prompt}\n\n{FORMAT_INSTRUCTIONS_EXTRACT}"

        # Profil LLM pour la longueur / température
        has_ctx = bool((ctx_for_prompt or "").strip())
        task_type = "chat_deep" if (use_memory or has_ctx) else "chat"
        profile = get_profile(task_type)

        case_id = getattr(req, "case_id", None)

        # max_tokens peut être surchargé par la requête, sinon on prend le profil
        max_tokens = getattr(req, "max_tokens", None) or profile.get("num_predict")
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))
        temperature = profile.get("temperature")

        logger.info(
            "chat: model=%s mode=%s case_id=%s use_memory=%s open_mode=%s history_len=%d ctx_len=%d max_tokens=%s",
            getattr(req, "model", None),
            getattr(req, "mode", None),
            getattr(req, "case_id", None),
            use_memory,
            open_mode,
            len(history),
            len(ctx_for_prompt or ""),
            max_tokens,
        )

        # --- Pseudonymisation éventuelle (noms / téléphones) pour modèles cloud ---
        final_prompt_sanitized, privacy_meta = anonymize_prompt_for_model(
            getattr(req, "model", None),
            final_prompt,
            case_id,
        )

        logger.info("chat: privacy meta=%s", {
            "sanitized": privacy_meta.get("sanitized"),
            "reason": privacy_meta.get("reason"),
            "orig_len": privacy_meta.get("original_len"),
            "san_len": privacy_meta.get("sanitized_len"),
            "orig_hash": privacy_meta.get("original_hash"),
            "san_hash": privacy_meta.get("sanitized_hash"),
            "repl": privacy_meta.get("replacements"),
            "case_id": case_id,
        })

        final_prompt = final_prompt_sanitized

        # Appel robuste à generate
        answer = _safe_generate(
            generate,
            req.model,
            final_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # --- Dé-pseudonymisation côté utilisateur ---
        try:
            text = answer or ""
            de = deanonymize_text_for_case(text, case_id)
            if de == text and case_id:
                de = deanonymize_text_for_case(text, None)
            answer = de
        except Exception:
            logger.exception("chat: deanonymize_text_for_case failed")

        return {"answer": answer, "sources": sources}

    async def chat_stream(self, req) -> StreamingResponse:
        """Requête de chat avec streaming."""
        history = getattr(req, "history", None) or []
        user_msg = getattr(req, "message", "") or ""
        use_memory = self._use_memory(req)

        # 1) Préparer le contexte brut (doc_text ou RAG)
        ctx_raw, sources = self._prepare_context(req)

        # 2) Construire le contexte final comme dans chat()
        if use_memory:
            ctx_for_prompt = self._build_context_with_history(
                ctx_raw=ctx_raw,
                history=history,
                user_msg=user_msg,
                use_memory=True,
            )
        else:
            ctx_for_prompt = ctx_raw

        mode = getattr(req, "mode", None) or "plain"

        open_mode = (not use_memory) and not (ctx_raw or "").strip()

        if open_mode:
            final_prompt = self._build_open_prompt(user_msg=user_msg, history=history)
        else:
            base_prompt = build_prompt(
                ctx_for_prompt,
                user_msg,
                lite=(not use_memory),
            )
            final_prompt = base_prompt

            if getattr(req, "system", None):
                final_prompt = req.system.strip() + "\n\n" + final_prompt

            if mode in ("extract", "extraction"):
                final_prompt = f"{final_prompt}\n\n{FORMAT_INSTRUCTIONS_EXTRACT}"

        # Profil LLM
        has_ctx = bool((ctx_for_prompt or "").strip())
        task_type = "chat_deep" if (use_memory or has_ctx) else "chat"
        profile = get_profile(task_type)

        case_id = getattr(req, "case_id", None)

        max_tokens = getattr(req, "max_tokens", None) or profile.get("num_predict")
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))
        temperature = profile.get("temperature")

        logger.info(
            "chat_stream: model=%s mode=%s case_id=%s use_memory=%s open_mode=%s history_len=%d ctx_len=%d max_tokens=%s",
            getattr(req, "model", None),
            getattr(req, "mode", None),
            getattr(req, "case_id", None),
            use_memory,
            open_mode,
            len(history),
            len(ctx_for_prompt or ""),
            max_tokens,
        )

        # --- Anonymisation une seule fois (avant appel modèle) ---
        final_prompt_sanitized, privacy_meta = anonymize_prompt_for_model(
            getattr(req, "model", None),
            final_prompt,
            case_id,
        )
        logger.debug("chat_stream: privacy_meta=%r", privacy_meta)

        def event_gen():
            yield sse({"type": "start"})
            seen = set()

            accumulated = ""
            sent_len = 0

            try:
                try:
                    stream_iter = _safe_stream_generate(
                        stream_generate,
                        req.model,
                        final_prompt_sanitized,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    logger.exception("chat_stream: stream_generate signature incompatible: %s", e)
                    yield sse({"type": "error", "message": f"stream backend signature error: {e}"})
                    yield sse({"type": "end"})
                    return

                source_index = {}
                try:
                    for s in sources or []:
                        src_name = s.get("source")
                        cid = s.get("chunk_id")
                        if src_name is None or cid is None:
                            continue
                        source_index[(src_name, cid)] = s
                except Exception:
                    source_index = {}

                for raw_delta in stream_iter:
                    if not raw_delta:
                        continue

                    accumulated += raw_delta

                    try:
                        de = deanonymize_text_for_case(accumulated, case_id)
                        if de == accumulated and case_id:
                            de = deanonymize_text_for_case(accumulated, None)
                    except Exception:
                        logger.exception("chat_stream: deanonymize_text_for_case failed on accumulated text")
                        de = accumulated

                    if sent_len < len(de):
                        new_part = de[sent_len:]
                        sent_len = len(de)
                    else:
                        new_part = ""

                    if not new_part:
                        continue

                    for m in self.cite_pattern.finditer(new_part):
                        src = (m.group(1) or "").strip()
                        try:
                            ch = int(m.group(2))
                        except Exception:
                            ch = None

                        key = (src, ch)
                        if key in seen:
                            continue
                        seen.add(key)

                        base_evt = {"type": "cite", "source": src, "chunk_id": ch}
                        meta = source_index.get(key) or {}
                        full_evt = {**base_evt, **meta}
                        yield sse(full_evt)

                    yield sse({"type": "token", "content": new_part})

                yield sse({"type": "end", "sources": sources})
            except Exception as e:
                logger.exception("chat_stream: error during streaming")
                yield sse({"type": "error", "message": f"stream error: {e}"})
                yield sse({"type": "end"})

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ---------- logique de contexte / mémoire ----------

    def _use_memory(self, req) -> bool:
        if getattr(req, "no_memory", False):
            return False
        if getattr(req, "mode", None) in ("extract", "extraction"):
            return False
        case_id = getattr(req, "case_id", None)
        if not case_id or case_id in ("0", "null"):
            return False
        return True

    def _prepare_context(self, req) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Prépare le contexte pour une requête donnée.

        Priorité :
        1) doc_text/document_text → contexte direct
        2) sinon affaire + mémoire → RAG
        """
        doc_text = getattr(req, "doc_text", None) or getattr(req, "document_text", None)
        if doc_text:
            text = str(doc_text)
            inv = _query_looks_like_inventory(getattr(req, "message", "") or "")
            doc_cap = MAX_DOC_CONTEXT_CHARS_INVENTORY if inv else MAX_DOC_CONTEXT_CHARS
            if len(text) > doc_cap:
                logger.info("Doc_text tronqué de %d à %d caractères pour le chat", len(text), doc_cap)
                text = text[:doc_cap]
            return text, []

        raw_context: List[str] = []
        sources: List[Dict[str, Any]] = []

        case_id = getattr(req, "case_id", None)
        user_msg = getattr(req, "message", "") or ""

        inv = getattr(req, "inventory_mode", None)
        if inv is None:
            inv = _query_looks_like_inventory(user_msg)
        inv = bool(inv)
        rag_cap = MAX_RAG_CONTEXT_CHARS_INVENTORY if inv else MAX_RAG_CONTEXT_CHARS
        doc_cap = MAX_DOC_CONTEXT_CHARS_INVENTORY if inv else MAX_DOC_CONTEXT_CHARS

        if case_id and case_id not in ("null", "0"):
            rag_enabled = os.getenv("RAG_ENABLED", "1") == "1"
            if rag_enabled and self._use_memory(req):
                try:
                    logger.info("Récupération RAG pour la requête de chat (group_by_unit=%s, inventory=%s)", True, inv)

                    # ✅ On récupère par UNITÉ (acte) si possible
                    retrieved = retrieve_hybrid(
                        query=user_msg,
                        case_id=case_id,
                        max_doc_chars=doc_cap,
                        max_rag_chars=rag_cap,
                        user_message_history=getattr(req, "history", None),
                        group_by_unit=True,
                        inventory_mode=inv,
                        extra_neighbors=int(os.getenv("CHAT_EXTRA_NEIGHBORS", "24")),
                    )

                    # retrieval peut renvoyer :
                    # - List[Dict group] (group_by_unit)
                    # - OU l'ancien format List[Dict docs]
                    context_blocks, srcs = self._build_rag_context_and_sources(retrieved, doc_cap)
                    raw_context.extend(context_blocks)
                    sources.extend(srcs)

                    logger.info("RAG a construit %d blocs contexte, %d sources", len(context_blocks), len(srcs))
                except Exception:
                    logger.exception("Échec de la récupération RAG")

        full_context = "\n\n".join([c for c in raw_context if c]).strip()

        if len(full_context) > doc_cap:
            logger.info("Contexte RAG tronqué de %d à %d caractères", len(full_context), doc_cap)
            full_context = full_context[-doc_cap:]

        return full_context, sources

    def _build_rag_context_and_sources(
        self,
        retrieved: Any,
        doc_cap: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Transforme les résultats retrieval en :
        - blocs de contexte (avec headers citables)
        - liste de sources riches pour le front
        """
        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []

        if not isinstance(retrieved, list):
            return context_blocks, sources

        # Cas 1 : group_by_unit → chaque item a "chunks"
        is_grouped = bool(retrieved and isinstance(retrieved[0], dict) and "chunks" in retrieved[0])

        if is_grouped:
            for g in retrieved:
                if not isinstance(g, dict):
                    continue

                src = g.get("source") or ""
                unit_title = g.get("unit_title") or ""
                unit_type = g.get("unit_type") or ""
                unit_id = g.get("unit_id")
                page_start = g.get("page_start")
                page_end = g.get("page_end")

                chunks = g.get("chunks") or []
                if not chunks:
                    continue

                # on choisit d’abord les ancres si présentes
                anchors = [c for c in chunks if isinstance(c, dict) and c.get("is_unit_anchor")]
                others = [c for c in chunks if isinstance(c, dict) and not c.get("is_unit_anchor")]

                chosen: List[Dict[str, Any]] = []
                if PREFER_UNIT_ANCHORS and anchors:
                    chosen.extend(anchors[:1])  # 1 ancre suffit
                # puis quelques chunks “preuve”
                chosen.extend(others[:MAX_CHUNKS_PER_UNIT_IN_CONTEXT])

                # header unité (pour guider le LLM)
                header = (
                    "=== ACTE / UNITÉ ===\n"
                    f"SOURCE: {src}\n"
                    f"UNIT_TITLE: {unit_title}\n"
                    f"UNIT_TYPE: {unit_type}\n"
                    f"UNIT_ID: {unit_id or ''}\n"
                    f"PAGES_UNIT: {page_start or ''}-{page_end or ''}\n"
                    "====================\n"
                )

                block_parts = [header]
                used_chars = 0

                for c in chosen:
                    text = (c.get("text") or "").strip()
                    if not text:
                        continue

                    page = c.get("page")
                    chunk_id = c.get("chunk_id")

                    # ✅ balise de citation compatible avec ton streaming
                    cite = f"[source: {src}, chunk {chunk_id}]"
                    subheader = f"{cite} page={page}\n"

                    piece = subheader + text + "\n\n"

                    if used_chars + len(piece) > doc_cap:
                        break
                    used_chars += len(piece)
                    block_parts.append(piece)

                    # sources front (riches)
                    meta = c.get("meta") or {}
                    sources.append({
                        "source": src,
                        "page": page,
                        "chunk_id": chunk_id,
                        "local_chunk_id": c.get("local_chunk_id"),
                        "unit_id": unit_id,
                        "unit_title": unit_title,
                        "unit_type": unit_type,
                        "page_start": page_start,
                        "page_end": page_end,
                        "is_unit_anchor": bool(c.get("is_unit_anchor")),
                        "is_fact": bool(c.get("is_fact")),
                        "doc_hash": meta.get("doc_hash"),
                        "url": meta.get("url"),
                    })

                context_blocks.append("".join(block_parts).strip())

            return context_blocks, sources

        # Cas 2 : ancien format : list[doc dict]
        for doc in retrieved:
            if not isinstance(doc, dict):
                continue
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            src = doc.get("source") or ""
            page = doc.get("page")
            chunk_id = doc.get("chunk_id")
            meta = doc.get("meta") or {}

            cite = f"[source: {src}, chunk {chunk_id}]"
            block = f"{cite} page={page}\n{text}"
            context_blocks.append(block)

            sources.append({
                "source": src,
                "page": page,
                "chunk_id": chunk_id,
                "local_chunk_id": doc.get("local_chunk_id"),
                "doc_hash": meta.get("doc_hash"),
                "url": meta.get("url"),
            })

        return context_blocks, sources

    def _build_context_with_history(
        self,
        ctx_raw: str,
        history: List[Any],
        user_msg: str,
        use_memory: bool,
    ) -> str:
        if not ctx_raw and not history:
            return ""

        ctx_parts: List[str] = []
        if ctx_raw:
            ctx_parts.append(ctx_raw + "\n\n")

        if history and use_memory:
            try:
                budget = int(os.getenv("HISTORY_BUDGET_TOKENS", "4096"))
            except Exception:
                budget = 4096

            used = 0

            for m in history:
                if hasattr(m, "role"):
                    role = m.role
                    content = m.content
                elif isinstance(m, dict):
                    role = m.get("role")
                    content = m.get("content")
                else:
                    continue

                content = (content or "").strip()
                if not content:
                    continue

                if role == "user":
                    prefix = "Utilisateur : "
                elif role == "assistant":
                    prefix = "Assistant : "
                else:
                    prefix = "Système : "

                chunk = f"{prefix}{content}\n"
                need = self._approx_tokens(chunk)

                if used + need > budget:
                    break

                ctx_parts.append(chunk)
                used += need

        def _history_last_content(hist):
            if not hist:
                return ""
            last = hist[-1]
            try:
                if isinstance(last, dict):
                    return (last.get("content", "") or "").strip()
                if hasattr(last, "content"):
                    return (getattr(last, "content") or "").strip()
            except Exception:
                pass
            try:
                return str(last).strip()
            except Exception:
                return ""

        last_content = _history_last_content(history)
        if user_msg and (not history or user_msg.strip() != last_content):
            ctx_parts.append(f"Utilisateur : {user_msg}\n")

        return "".join(ctx_parts).strip()

    # ---------- open prompt factorisation ----------
    def _build_open_prompt(self, user_msg: str, history: List[Any]) -> str:
        simple = (user_msg or "").strip().lower()
        simple = re.sub(r"^[^\w]+", "", simple)

        investigative_keywords = (
            "enquêt",
            "procès-verbal",
            "pv ",
            "gendarmerie",
            "police",
            "audition",
            "témoin",
            "victime",
            "mis en cause",
            "garde à vue",
            "instruction",
            "réquisition",
            "perquisition",
            "commission rogatoire",
            "dossier",
        )
        is_investigative = any(kw in simple for kw in investigative_keywords)

        text_to_summarize: Optional[str] = None
        if history and ("résum" in simple or "resum" in simple):
            for m in reversed(history):
                if hasattr(m, "role"):
                    role = m.role
                    content = m.content
                elif isinstance(m, dict):
                    role = m.get("role")
                    content = m.get("content")
                else:
                    continue
                if (role or "").strip() == "assistant":
                    candidate = (content or "").strip()
                    if candidate:
                        text_to_summarize = candidate
                        break

            if not text_to_summarize:
                for m in reversed(history):
                    if hasattr(m, "role"):
                        role = m.role
                        content = m.get("content") if isinstance(m, dict) else m.content
                    elif isinstance(m, dict):
                        role = m.get("role")
                        content = m.get("content")
                    else:
                        continue
                    if (role or "").strip() == "user":
                        candidate = (content or "").strip()
                        if candidate:
                            text_to_summarize = candidate
                            break

        if text_to_summarize:
            return (
                "Tu es Axion, un assistant IA francophone polyvalent.\n"
                "On te fournit un texte que tu dois résumer de façon claire, concise et fidèle.\n"
                "Si le texte concerne une enquête, un dossier ou des éléments judiciaires, "
                "tu peux, si c'est utile, mettre en évidence les faits importants "
                "(personnes, dates, lieux, actions principales), mais ce n'est pas obligatoire.\n\n"
                "Texte à résumer :\n"
                f"{text_to_summarize}\n\n"
                "Résumé :"
            )

        history_block = self._build_history_block(history)
        if is_investigative:
            return (
                "Tu es Axion, un assistant IA francophone polyvalent, avec une forte expertise en analyse "
                "d'enquêtes judiciaires pénales et de dossiers complexes.\n"
                "Tu réponds de manière naturelle et fluide, en exploitant pleinement tes capacités de raisonnement.\n"
                "Pour les questions qui relèvent manifestement d'une enquête ou d'un dossier, "
                "tu peux structurer ta réponse comme un analyste judiciaire expérimenté "
                "(faits, analyse, points d'attention, limites), sans t'autocensurer inutilement.\n\n"
                + (history_block or "")
                + "Question de l'utilisateur :\n"
                + (user_msg or "")
            )

        return (
            "Tu es un assistant IA francophone polyvalent.\n"
            "Tu peux répondre librement à des questions sur tous les sujets en exploitant pleinement "
            "tes capacités de réflexion, d'analyse et de créativité.\n\n"
            + (history_block or "")
            + "Question de l'utilisateur :\n"
            + (user_msg or "")
        )

    # ---------- diagnostics / outils ----------

    def diagnose(self, req) -> dict:
        history = getattr(req, "history", None) or []
        use_memory = self._use_memory(req)

        ctx_raw, _sources = self._prepare_context(req)
        ctx_for_prompt = self._build_context_with_history(
            ctx_raw=ctx_raw,
            history=history,
            user_msg=getattr(req, "message", ""),
            use_memory=use_memory,
        )

        return {
            "case_id": getattr(req, "case_id", None),
            "use_memory": use_memory,
            "history_len": len(history),
            "ctx_raw_len": len(ctx_raw),
            "ctx_for_prompt_len": len(ctx_for_prompt),
            "max_doc_context_chars": MAX_DOC_CONTEXT_CHARS,
            "max_rag_context_chars": MAX_RAG_CONTEXT_CHARS,
            "rag_enabled": os.getenv("RAG_ENABLED", "1") == "1",
        }


# Helpers robustes pour appeler generate / stream_generate en tolérant différentes signatures
def _safe_generate(func, model, prompt, temperature=None, max_tokens=None):
    attempts = [
        dict(temperature=temperature, max_tokens=max_tokens),
        dict(temperature=temperature),
        dict(max_tokens=max_tokens),
        {},
    ]
    last_exc = None
    for kw in attempts:
        try:
            call_kw = {k: v for k, v in kw.items() if v is not None}
            return func(model, prompt, **call_kw)
        except TypeError as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    return None


def _safe_stream_generate(func, model, prompt, temperature=None, max_tokens=None):
    attempts = [
        dict(temperature=temperature, max_tokens=max_tokens),
        dict(temperature=temperature, max_length=max_tokens),
        dict(temperature=temperature),
        dict(),
    ]
    last_exc = None
    for kw in attempts:
        try:
            call_kw = {k: v for k, v in kw.items() if v is not None}
            it = func(model, prompt, **call_kw)
            iter(it)
            return it
        except TypeError as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    return iter(())
