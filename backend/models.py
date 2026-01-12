# backend/models.py
# Schémas Pydantic pour les requêtes et réponses de l'API de chat.
# Auteur: Jax

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    doc_text: Optional[str] = None
    case_id: Optional[str] = None
    model: Optional[str] = None
    use_memory: Optional[bool] = False

    # mode : "", "memory", "memoire", "context", "document", "upload", etc.
    mode: Optional[str] = Field(None, description="Mode de fonctionnement du chat")

    # ✅ max tokens de sortie souhaités
    # On aligne le plafond sur OLLAMA_NUM_PREDICT (12k/16k) pour permettre des réponses “inventaire”.
    max_tokens: Optional[int] = Field(
        None,
        ge=64,
        le=16384,
        description="Nombre max de tokens de réponse",
    )

    # system prompt optionnel envoyé par le front
    system: Optional[str] = Field(
        None,
        description="Instruction système additionnelle",
    )

    # Historique de la conversation (tour précédent, etc.)
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Historique (user/assistant/system) de la conversation",
    )

    # Texte brut envoyé (upload ou copier-coller)
    document_text: Optional[str] = Field(
        default=None,
        description="Alias de doc_text si utilisé ailleurs",
    )

    # 👉 NOUVEAU : identifiant d'un document de session (upload hors affaire)
    doc_session_id: Optional[str] = Field(
        default=None,
        description="Identifiant d'un document de session uploadé (hors affaire)",
    )

    # ✅ (optionnel) Flags avancés : permettent au front de piloter le RAG
    # Même si chat_service.py auto-détecte inventory, c’est pratique pour forcer un mode.
    inventory_mode: Optional[bool] = Field(
        default=None,
        description="Force le mode inventaire/exhaustif (RAG coverage-first)",
    )

    group_by_unit: Optional[bool] = Field(
        default=None,
        description="Retour retrieval groupé par unité/acte (PV) si supporté",
    )


class ChatResponse(BaseModel):
    answer: str
    # sources RAG (liste de meta)
    sources: Optional[list] = None
