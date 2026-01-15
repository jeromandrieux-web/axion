// static/js/chat-upload.js
// Chargement de documents vers /api/chat/upload.
// Le texte extrait est stocké dans l’état Alpine (currentDocText) pour
// être utilisé comme contexte lors des appels suivants à /api/chat/stream.
// Auteur : Jax

(function () {
  "use strict";

  console.log("[axion] chat-upload init (doc_text + contexte)");

  function getRoot() {
    const body = document.querySelector("body");
    if (!body || !body._x_dataStack || !body._x_dataStack[0]) return null;
    return body._x_dataStack[0];
  }

  function makeAssistantBubble(root, html) {
    const id = crypto.randomUUID();
    root.messages = root.messages || [];
    root.messages.push({
      id,
      role: "assistant",
      html,
      sources: [],
      typing: false,
    });
    try {
      const scroll = document.getElementById("scrollArea");
      if (scroll) scroll.scrollTop = scroll.scrollHeight;
    } catch (_) {}
    return id;
  }

  async function handleFileDrop(file) {
    if (!file) return;

    const root = getRoot();
    if (!root) {
      alert("Interface non initialisée (Alpine).");
      return;
    }

    // Message "user" minimal : indique juste quel fichier est chargé
    root.messages = root.messages || [];
    root.messages.push({
      id: crypto.randomUUID(),
      role: "user",
      html: "📄 " + root.escape(file.name),
    });

    // Appel backend : extraction du texte (AUCUNE réponse LLM ici)
    const fd = new FormData();
    fd.append("file", file);

    let resp;
    try {
      resp = await fetch("/api/chat/upload", {
        method: "POST",
        body: fd,
      });
    } catch (e) {
      console.error("[axion] erreur réseau /api/chat/upload", e);
      makeAssistantBubble(
        root,
        '<span class="text-red-400">Erreur réseau lors du chargement du document.</span>'
      );
      return;
    }

    if (!resp.ok) {
      let txt = "";
      try {
        txt = await resp.text();
      } catch (_) {}
      console.error(
        "[axion] /api/chat/upload HTTP",
        resp.status,
        resp.statusText,
        txt
      );
      makeAssistantBubble(
        root,
        '<span class="text-red-400">Erreur lors du chargement du document ('
          + resp.status +
          ').</span>'
      );
      return;
    }

    let data = null;
    try {
      data = await resp.json();
    } catch (e) {
      console.error("[axion] réponse non JSON de /api/chat/upload", e);
    }

    if (!data || data.ok === false) {
      const errMsg =
        (data && data.error) ||
        "Impossible d’extraire du texte exploitable de ce document.";
      console.warn("[axion] upload: échec extraction", errMsg);
      makeAssistantBubble(
        root,
        '<span class="text-red-400">'
          + root.escape(errMsg) +
          "</span>"
      );
      return;
    }

    // ✅ On stocke le texte extrait dans l'état Alpine
    try {
      root.currentDocText = data.doc_text || "";
      root.currentDocName = data.filename || file.name || "document";
      console.log(
        "[axion] document chargé pour le chat (chars=",
        (root.currentDocText || "").length,
        ")"
      );
    } catch (e) {
      console.warn("[axion] impossible de stocker currentDocText", e);
    }
  }

  document.addEventListener(
    "dragover",
    (e) => {
      e.preventDefault();
      e.stopPropagation();
    },
    false
  );

  document.addEventListener(
    "drop",
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) {
        handleFileDrop(files[0]);
      }
    },
    false
  );
})();
