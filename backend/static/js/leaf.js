// static/js/leaf.js
// affiche la progression dans le panneau "Outils & Synthèse"
// gère la synthèse thématique via /api/leaf/stream
// Auteur: Jax

(function () {
  function getRoot() {
    const body = document.querySelector("body");
    return (body && body._x_dataStack && body._x_dataStack.length)
      ? body._x_dataStack[0]
      : null;
  }

  function showLeafProgress(label) {
    const box = document.getElementById("leaf-progress");
    const txt = document.getElementById("leaf-progress-text");
    const fill = document.getElementById("leaf-progress-fill");
    if (box) box.style.display = "";
    if (txt) txt.textContent = label || "Synthèse en cours…";
    if (fill) fill.style.width = "5%";
  }

  function setLeafProgress(value, label) {
    const txt = document.getElementById("leaf-progress-text");
    const fill = document.getElementById("leaf-progress-fill");
    const pct = Math.min(100, Math.max(0, Math.round(value * 100)));
    if (fill) fill.style.width = pct + "%";
    if (txt) txt.textContent = label || "";
  }

  function hideLeafProgress() {
    const box = document.getElementById("leaf-progress");
    if (box) box.style.display = "none";
  }

  async function ssePost(url, payload, onEvent) {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "accept": "text/event-stream",
      },
      body: JSON.stringify(payload),
    });

    if (!resp.ok || !resp.body) {
      onEvent({ type: "error", message: "HTTP " + resp.status });
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        let ev;
        try {
          ev = JSON.parse(jsonStr);
        } catch (e) {
          onEvent({ type: "error", message: "JSON invalide reçu" });
          continue;
        }
        onEvent(ev);
      }
    }

    onEvent({ type: "end" });
  }

  async function runLeaf() {
    const root = getRoot();
    const btn = document.getElementById("summary-btn");
    const link = document.getElementById("leaf-download");

    const caseId = (root && root.useMemory) ? (root.caseId || "") : "";
    const model = root ? (root.model || null) : null;

    if (!caseId) {
      alert("Sélectionne d'abord une affaire dans 📁 Affaires.");
      return;
    }

    if (btn) btn.disabled = true;
    if (link) {
      link.style.display = "none";
      link.removeAttribute("href");
    }

    showLeafProgress("Démarrage de la synthèse…");

    await ssePost("/api/leaf/stream", { case_id: caseId, model }, (ev) => {
      if (!ev || !ev.type) return;

      if (ev.type === "start") {
        setLeafProgress(0.05, "Analyse du dossier…");
        return;
      }

      if (ev.type === "stage") {
        setLeafProgress(0.2, ev.message || "Étape…");
        return;
      }

      if (ev.type === "progress") {
        const v = typeof ev.value === "number" ? ev.value : 0;
        const lbl = ev.stage ? ("Étape : " + ev.stage) : "Synthèse en cours…";
        setLeafProgress(v, lbl);
        return;
      }

      if (ev.type === "error") {
        setLeafProgress(1, "Erreur : " + (ev.message || "inconnue"));
        if (btn) btn.disabled = false;
        return;
      }

      if (ev.type === "end") {
        // on affiche le lien
        if (link) {
          link.href = `/api/leaf/${encodeURIComponent(caseId)}/download`;
          link.style.display = "";
        }
        setLeafProgress(1, "Synthèse terminée ✅");
        if (btn) btn.disabled = false;
        return;
      }

      // on ignore ev.type === "text"
    });
  }

  function bind() {
    const btn = document.getElementById("summary-btn");
    if (!btn) return;
    btn.addEventListener("click", function () {
      runLeaf().catch(console.error);
    });

    // on expose quand même au cas où
    window.makeLeaf = runLeaf;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
