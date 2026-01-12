// static/js/ingest.js
// gère l'upload + SSE + mise à jour Alpine + fallback DOM
// Auteur: Jax

(function () {
  function getRoot() {
    const body = document.querySelector("body");
    if (body && body._x_dataStack && body._x_dataStack.length) {
      return body._x_dataStack[0];
    }
    return {};
  }

  function ensureState(root) {
    if (typeof root.showProgress === "undefined") root.showProgress = false;
    if (typeof root.progress === "undefined") root.progress = 0;
    if (typeof root.phase === "undefined") root.phase = "";
    if (typeof root.task === "undefined") root.task = "";
    if (typeof root.eta === "undefined") root.eta = "";
    if (typeof root.ocrPages === "undefined") root.ocrPages = 0;
  }

  function hideDomProgress() {
    // 🔧 au cas où Alpine n'obéit pas, on masque en dur
    const el = document.getElementById("ingest-progress");
    if (el) {
      el.style.display = "none";
    }
  }

  function showDomProgress() {
    const el = document.getElementById("ingest-progress");
    if (el) {
      el.style.display = "";
    }
  }

  function parseSSELine(line) {
    if (!line) return null;
    if (line.startsWith("data:")) {
      const raw = line.slice(5).trim();
      try {
        return JSON.parse(raw);
      } catch (e) {
        return { type: "raw", raw };
      }
    }
    return null;
  }

  function applyEvent(ev) {
    const root = getRoot();
    ensureState(root);

    if (!ev || !ev.type) return;

    if (ev.type === "start") {
      root.showProgress = true;
      root.progress = 2;
      root.phase = "Préparation des fichiers…";
      root.task = ev.total_files ? `Fichiers : ${ev.total_files}` : "Analyse…";
      showDomProgress();
      return;
    }

    if (ev.type === "file") {
      root.showProgress = true;
      root.phase = "Lecture et découpe";
      root.task = ev.file ? `Fichier : ${ev.file}` : "Fichier…";
      showDomProgress();
      return;
    }

    if (ev.type === "progress") {
      root.showProgress = true;
      root.phase = "Traitement…";
      if (typeof ev.percent === "number") {
        root.progress = Math.max(0, Math.min(100, ev.percent));
      } else if (ev.current && ev.total) {
        root.progress = Math.round((ev.current / ev.total) * 100);
      }
      if (ev.file) root.task = `Traitement de ${ev.file}`;
      showDomProgress();
      return;
    }

    if (ev.type === "done") {
      root.showProgress = true;
      root.progress = 100;
      // IMPORTANT : on met le texte dans task (ton index fait phase || task)
      root.phase = "";
      root.task = "Ingestion terminée ✅";
      root.ocrPages = ev.ocr_pages || 0;

      setTimeout(() => {
        const r = getRoot();
        if (r) {
          r.showProgress = false;
          r.task = "";
          r.phase = "";
        }
        hideDomProgress();
      }, 1500);
      return;
    }

    if (ev.type === "end") {
      // certains backends envoient "end" à la toute fin
      const r = getRoot();
      if (r) {
        if (r.progress < 100) r.progress = 100;
        r.phase = "";
        r.task = "Ingestion terminée ✅";
      }
      setTimeout(() => {
        const r2 = getRoot();
        if (r2) {
          r2.showProgress = false;
          r2.task = "";
          r2.phase = "";
        }
        hideDomProgress();
      }, 1500);
      return;
    }

    if (ev.type === "error") {
      root.showProgress = true;
      root.phase = "Erreur";
      root.task = ev.message || "Erreur pendant l’ingestion";
      showDomProgress();
      // on ne masque pas tout de suite pour que l'user voie l'erreur
      return;
    }
  }

  async function ingest() {
    const root = getRoot();
    ensureState(root);

    const filesInput = document.getElementById("files");
    const files = filesInput && filesInput.files ? filesInput.files : [];

    const caseId = root && root.caseId ? root.caseId : (document.getElementById("case_id")?.value || "");

    if (!caseId) {
      alert("Choisis d'abord une affaire.");
      return;
    }
    if (!files.length) {
      alert("Sélectionne au moins un fichier.");
      return;
    }

    root.showProgress = true;
    root.progress = 1;
    root.phase = "Envoi des fichiers…";
    root.task = "Upload…";
    showDomProgress();

    const fd = new FormData();
    fd.append("case_id", caseId);
    for (let i = 0; i < files.length; i++) {
      fd.append("files", files[i], files[i].name);
    }

    const resp = await fetch("/api/ingest/stream", {
      method: "POST",
      body: fd,
      headers: { "Accept": "text/event-stream" },
    });

    if (!resp.ok || !resp.body) {
      root.phase = "Erreur";
      root.task = `HTTP ${resp.status}`;
      showDomProgress();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trimEnd();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        if (line.startsWith("event:")) continue;
        const ev = parseSSELine(line);
        if (ev) applyEvent(ev);
      }
    }

    // sécurité au cas où on n'a pas reçu end/done
    setTimeout(() => {
      const r = getRoot();
      if (r) {
        r.showProgress = false;
        r.task = "";
        r.phase = "";
      }
      hideDomProgress();
    }, 2000);
  }

  // exposé global
  window.ingest = ingest;
})();
