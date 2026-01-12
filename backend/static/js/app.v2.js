// app.v2.js - Application principale Alpine pour Axion
// Parseur SSE générique + fallback ingest + logique Alpine
// initialise l'application alpine "chatApp"
// gère le chat, l'ingestion, la synthèse LEAF et chrono, etc.

"use strict";

// ==========================
// petit util SSE générique (tolérant event: ... \n data: ...)
// ==========================
async function parseSSEStream(reader, onEvent) {
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!raw) continue;

      // On accepte les blocs avec ou sans "event:" ; on cherche la ligne "data:"
      const dataLine = raw.split("\n").find((ln) => ln.startsWith("data:"));
      if (!dataLine) continue;
      const jsonStr = dataLine.slice(5).trim();
      try {
        const evt = JSON.parse(jsonStr);
        onEvent && onEvent(evt);
      } catch (e) {
        // ignore
      }
    }
  }
}

// on l'expose pour chat-upload.js et d'autres scripts
window.parseSSEStream = parseSSEStream;

// ==========================
// fallback ingest 
// ==========================
async function startIngestStream(formData, onEvent) {
  // si ton ancien script ingest.js existe, on le garde
  if (window.streamIngest && typeof window.streamIngest === "function") {
    return window.streamIngest(formData, onEvent);
  }
  const resp = await fetch("/api/ingest/stream", {
    method: "POST",
    body: formData,
  });
  if (!resp.ok) throw new Error("Upload failed: " + resp.status);
  if (!resp.body) throw new Error("Streaming non supporté");
  const reader = resp.body.getReader();
  await parseSSEStream(reader, onEvent);
}

// ==========================
// l’appli Alpine
// ==========================
function chatApp() {
  return {
    // --- état global UI ---
    models: [],
    model: "",
    useMemory: false,
    caseId: "",
    cases: [],
    docs: [],

    // --- chat ---
    messages: [],
    message: "",
    sending: false,

    // --- progress partagé (ingest / leaf / chrono) ---
    showProgress: false,
    busy: false,
    task: "",
    progress: 0,
    eta: null,
    phase: "",
    ocrPages: 0,
    progressKind: null,

    // --- aide / panneaux ---
    showHelp: false,

    // --- pensées (si un jour tu réactives le "think") ---
    think: { show: false, sticky: false, text: "", inBlock: false, buf: "", openTag: null },

    // --- analyse / extraction ---
    extractTheme: "phones",
    extractPerson: "",
    extracting: false,

    async extractPhones() {
      this.extracting = true;
      const outEl = document.getElementById("extractOutputPhones");
      if (outEl) {
        outEl.innerHTML = this.renderAnswer("Extraction en cours…");
      }

      try {
        const r = await fetch("/api/extract/phones", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            case_id: this.caseId,
            theme: this.extractTheme
            // autres paramètres éventuels…
          }),
        });

        const data = await r.json().catch(() => ({}));
        const md = data.result || data.text || "";

        if (outEl) {
          outEl.innerHTML = this.renderAnswer(md);
        }
      } catch (e) {
        console.error("extractPhones error", e);
        if (outEl) {
          outEl.innerHTML = '<span class="text-red-400">Erreur extraction téléphones.</span>';
        }
      } finally {
        this.extracting = false;
      }
    },

    // --- contexte document uploadé ---
    currentDocText: "",
    currentDocName: "",

    // --- LEAF ---
    leafPages: 25,

    // --- init ---
    async init() {
      console.debug("[axion] chatApp.init()");
      try {
        await this.fetchModels();
      } catch (_) {}
      try {
        await this.refreshCases();
      } catch (_) {}

      this.$nextTick(() => {
        this.scrollBottom(true);
        this.setupAutoresize();
      });
    },

    // ==========================
    // chargement modèles / affaires
    // ==========================
    async fetchModels() {
      try {
        const r = await fetch("/api/models");
        const d = await r.json();
        let arr = [];
        let def = "";
        if (Array.isArray(d)) arr = d;
        else if (Array.isArray(d.models)) {
          arr = d.models;
          def = d.default || "";
        }
        this.models = arr;
        const saved = localStorage.getItem("axion_model");
        if (saved && arr.includes(saved)) this.model = saved;
        else this.model = def || arr[0] || "";
      } catch (e) {
        console.warn("fetchModels error", e);
        this.models = [];
        this.model = "";
      }
    },

    async refreshCases() {
      try {
        const r = await fetch("/api/cases");
        const d = await r.json();
        let arr = [];
        if (Array.isArray(d)) arr = d;
        else if (Array.isArray(d.cases)) arr = d.cases;
        else if (Array.isArray(d.items)) arr = d.items;
        arr = arr.map((c) =>
          typeof c === "string" ? c : c.case_id || c.id || c.name || String(c)
        );
        this.cases = arr;
        if (!arr.includes(this.caseId)) {
          this.caseId = arr[0] || "";
        }
        if (this.caseId) {
          await this.loadDocs();
        }
      } catch (e) {
        console.warn("refreshCases error", e);
        this.cases = [];
      }
    },

    async createCase() {
      const idEl = document.getElementById("newcase");
      const id = idEl ? idEl.value.trim() : "";
      if (!id) {
        alert("Saisis un ID d’affaire");
        return;
      }
      const fd = new FormData();
      fd.append("case_id", id);
      const r = await fetch("/api/case/create", { method: "POST", body: fd });
      const j = await r.json().catch(() => null);
      const ok = (j && (j.ok || j.success || j.created)) || r.ok;
      if (!ok) {
        alert((j && (j.error || j.message)) || "Erreur création affaire");
        return;
      }
      if (idEl) idEl.value = "";
      await this.refreshCases();
      this.caseId = id;
      await this.loadDocs();
      alert("Affaire créée");
    },

    async loadDocs() {
      if (!this.caseId) {
        this.docs = [];
        return;
      }
      try {
        const r = await fetch(
          "/api/case/docs?case_id=" + encodeURIComponent(this.caseId)
        );
        const j = await r.json().catch(() => ({}));
        let arr = [];
        if (Array.isArray(j.docs)) arr = j.docs;
        else if (Array.isArray(j.items)) arr = j.items;
        this.docs = arr.map((d) =>
          typeof d === "string" ? { name: d, hash: null, n_chunks: 0 } : d
        );
      } catch (e) {
        console.warn("loadDocs error", e);
        this.docs = [];
      }
    },

    // ==========================
    // util text / historique
    // ==========================
    escape(s) {
      return String(s).replace(/[&<>"']/g, (m) => {
        return {
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[m];
      });
    },
    nl2br(s) {
      return String(s).replaceAll("\n", "<br>");
    },
    renderAnswer(md) {
      try {
        // si déjà HTML, on le laisse ; sinon marked.parse convertit le Markdown
        const rawHtml = (typeof md === "string" && window.marked) ? window.marked.parse(md) : (md || "");
        const safeHtml = window.DOMPurify ? window.DOMPurify.sanitize(rawHtml) : rawHtml;
        return safeHtml;
      } catch (e) {
        return this.escape(String(md || ""));
      }
    },

    // convertit l'HTML stocké dans les messages en texte brut pour l'historique
    htmlToText(html) {
      if (!html) return "";
      try {
        const tmp = document.createElement("div");
        // on remet les <br> en \n pour garder un minimum de structure
        tmp.innerHTML = String(html).replace(/<br\s*\/?>/gi, "\n");
        const raw = tmp.textContent || tmp.innerText || "";
        return raw.replace(/\s+\n/g, "\n").replace(/\n\s+/g, "\n").trim();
      } catch (_) {
        // fallback simpliste
        return String(html).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
      }
    },

    buildHistoryPayload() {
      const maxMessages = 64;
      const msgs = this.messages || [];
      const out = [];

      const start = Math.max(0, msgs.length - maxMessages);
      for (let i = start; i < msgs.length; i++) {
        const m = msgs[i];
        if (!m) continue;
        if (m.role !== "user" && m.role !== "assistant") continue;

        const text = this.htmlToText(m.html || "");
        const content = text.trim();
        if (!content) continue;

        out.push({
          role: m.role,
          content,
        });
      }
      return out;
    },

    copyMsg(m) {
      try {
        const text = this.htmlToText(m.html || "");
        if (!text) return;

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text);
        } else {
          // fallback vieux navigateurs
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
        }

        // (optionnel) petit feedback console, ou toast plus tard
        console.debug("[axion] message copié dans le presse-papiers");
      } catch (e) {
        console.warn("copyMsg failed", e);
      }
    },

    // ==========================
    // chat
    // ==========================
    scrollBottom(force = false) {
      const el = document.getElementById("scrollArea");
      if (!el) return;
      if (force) {
        el.scrollTop = el.scrollHeight;
        return;
      }
      const near = el.scrollTop + el.clientHeight >= el.scrollHeight - 15;
      if (near) {
        el.scrollTop = el.scrollHeight;
      }
    },

    pushUser(text) {
      this.messages.push({
        id: crypto.randomUUID(),
        role: "user",
        // Préfixe emoji pour l'utilisateur
        html: "👤 " + this.nl2br(this.escape(text)),
      });
      this.$nextTick(() => this.scrollBottom(true));
    },

    pushAssistantPlaceholder() {
      const id = crypto.randomUUID();
      const avatarHtml =
        '<span class="msg-avatar-wrap typing thinking"><img src="/static/img/logo-chat.png" class="msg-avatar" alt="assistant"/></span>';

      const dotsHtml = '<span class="typing dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';

      this.messages.push({
        id,
        role: "assistant",
        avatarHtml,
        html: avatarHtml + " " + dotsHtml,
        typing: true,
        sources: [],
      });
      this.$nextTick(() => this.scrollBottom(false));
      return id;
    },

    updateAssistant(id, html, sources) {
      const m = this.messages.find((x) => x.id === id);
      if (!m) return;

      // Ne désactiver la flag typing que si l'appeleur l'a explicitement demandé
      // (la logique d'arrêt se fait depuis send() sur evt.type === "end"/"error")
      // Ici on met simplement à jour le contenu tout en conservant l'avatar wrapper
      const contentHtml = String(html || "");

      if (m.avatarHtml) {
        // Conserver le wrapper avatar (avec ou sans .typing selon m.avatarHtml)
        m.html = m.avatarHtml + " " + contentHtml;
      } else {
        // ancien comportement de secours
        const trimmed = contentHtml.trim();
        const startsWithAvatar = /^<img[^>]+class=(["'])?([^"'>]*\s)?msg-avatar(\s[^"'>]*)?\1/i.test(
          trimmed
        );
        if (startsWithAvatar) {
          m.html = contentHtml;
        } else {
          m.html = '<img src="/static/img/logo-chat.png" class="msg-avatar" alt="assistant"/> ' + contentHtml;
        }
      }

      if (sources && sources.length) {
        m.sources = m.sources || [];
        const exists = new Set(
          m.sources.map(
            (s) => `${s.source}#${s.page || ""}#${s.chunk_id || ""}`
          )
        );
        for (const s of sources) {
          const key = `${s.source}#${s.page || ""}#${s.chunk_id || ""}`;
          if (!exists.has(key)) m.sources.push(s);
        }
      }
      this.$nextTick(() => this.scrollBottom(false));
    },

    // ==========================
    // envoi message -> chat/stream
    // ==========================
    async send() {
      const text = this.message.trim();
      if (!text) return;
      if (!this.model) {
        alert("Choisis un modèle d’abord.");
        return;
      }
      this.sending = true;

      // 👉 on capture l'historique avant d'ajouter le nouveau message
      const historyPayload = this.buildHistoryPayload();

      this.pushUser(text);
      const aId = this.pushAssistantPlaceholder();
      let acc = "";

      const payload = {
        message: text,
        model: this.model,
        mode: this.useMemory ? "memory" : "plain",
        case_id: this.useMemory ? this.caseId : null,
        history: historyPayload, // ⬅️ mémoire conversationnelle envoyée au backend
      };

      // Si un document a été chargé pour ce chat, on l’envoie comme contexte explicite
      if (this.currentDocText && this.currentDocText.trim()) {
        payload.doc_text = this.currentDocText;
      }

      try {
        const r = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!r.body) throw new Error("streaming non supporté");
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        const handle = (evt) => {
          // si le backend renvoie directement le texte du document (doc_text),
          // on le stocke et on l'affiche comme réponse assistant
          if (evt && evt.doc_text) {
            this.currentDocText = String(evt.doc_text);
            this.currentDocName = evt.doc_name || "";
            this.updateAssistant(aId, this.renderAnswer(this.currentDocText));
            return;
          }
          if (evt.type === "token") {
            acc += evt.content || "";
            this.updateAssistant(aId, this.renderAnswer(acc));
          } else if (evt.type === "cite") {
            this.updateAssistant(aId, this.renderAnswer(acc), [evt]);
          } else if (evt.type === "status") {
            acc += "\n[" + (evt.message || "") + "]";
            this.updateAssistant(aId, this.renderAnswer(acc));
          } else if (evt.type === "end") {
            const mm = this.messages.find((x) => x.id === aId);
            if (mm && mm.avatarHtml) {
              mm.avatarHtml = mm.avatarHtml
                .replace(/\btyping\b/g, "")
                .replace(/\bthinking\b/g, "")
                .replace(/\s+/g, " ")
                .trim();
            }
            this.updateAssistant(
              aId,
              this.renderAnswer(acc),
              evt.sources || []
            );
          } else if (evt.type === "error") {
            const mm = this.messages.find((x) => x.id === aId);
            if (mm && mm.avatarHtml) {
              mm.avatarHtml = mm.avatarHtml
                .replace(/\btyping\b/g, "")
                .replace(/\bthinking\b/g, "")
                .replace(/\s+/g, " ")
                .trim();
            }
            this.updateAssistant(
              aId,
              '<span class="text-red-400">' +
                this.escape(evt.message || "Erreur") +
                "</span>"
            );
          }
        };

        const parse = () => {
          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const raw = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (!raw) continue;
            if (raw.startsWith("data:")) {
              const js = raw.slice(5).trim();
              try {
                const evt = JSON.parse(js);
                handle(evt);
              } catch (_) {}
            }
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          parse();
        }
        parse();
      } catch (e) {
        console.error("send /api/chat/stream error", e);
        this.updateAssistant(
          aId,
          '<span class="text-red-400">Erreur lors du streaming.</span>'
        );
      } finally {
        this.message = "";
        // recalculer la hauteur de la zone de saisie après reset
        this.$nextTick(() => this.autoresize());
        this.sending = false;
        this.$nextTick(() => this.scrollBottom(false));
      }
    },

    // ==========================
    // ingestion
    // ==========================
    async ingest() {
      const cid = this.caseId;
      if (!cid) {
        alert("Choisis d’abord une affaire.");
        return;
      }
      const files = document.getElementById("files").files;
      if (!files.length) {
        alert("Sélectionne des fichiers.");
        return;
      }

      this.task = "Ingestion en cours…";
      this.progress = 0;
      this.phase = "";
      this.eta = null;
      this.ocrPages = 0;
      this.progressKind = "ingest";
      this.showProgress = true;
      this.busy = true;

      const fd = new FormData();
      fd.append("case_id", cid);
      for (const f of files) fd.append("files", f);

      try {
        await startIngestStream(fd, (evt) => {
          if (evt.type === "progress") {
            this.progress = evt.percent || 0;
            this.eta = evt.eta || null;
            if (typeof evt.ocr_pages === "number") {
              this.ocrPages = evt.ocr_pages;
            }
          } else if (evt.type === "phase" || evt.type === "status") {
            this.phase = evt.label || evt.message || "";
          } else if (evt.type === "done" || evt.type === "end") {
            // 💡 fin normale signalée par le backend
            this.progress = 100;
            this.phase = "Ingestion terminée ✅";
            this.$nextTick(() => {
              setTimeout(() => {
                this.showProgress = false;
                if (this.progressKind === "ingest") this.progressKind = null;
              }, 800);
            });
            this.refreshCases();
            this.loadDocs();
          } else if (evt.type === "error") {
            alert("Erreur ingestion: " + (evt.message || ""));
            this.$nextTick(() => {
              setTimeout(() => {
                this.showProgress = false;
                if (this.progressKind === "ingest") this.progressKind = null;
              }, 1200);
            });
          }
        });
      } catch (e) {
        alert("Erreur pendant l’ingestion.");
      } finally {
        // 🔐 Sécurité : même si aucun event 'end' n'a été reçu,
        // on ferme proprement la barre de progression côté UI.
        this.busy = false;
        this.$nextTick(() => {
          // si le backend a fermé le flux sans envoyer "done/end"
          if (this.progress < 100 && this.showProgress && this.progressKind === "ingest") {
            this.progress = 100;
            if (!this.phase) {
              this.phase = "Ingestion terminée (flux fermé) ✅";
            }
            setTimeout(() => {
              this.showProgress = false;
              this.progressKind = null;
            }, 900);
          }
        });
      }
    },

    // ==========================
    // synthèse LEAF
    // ==========================
    async makeLeaf() {
      const cid = this.caseId;
      if (!cid) {
        alert("Aucune affaire sélectionnée");
        return;
      }

      this.task = "Synthèse LEAF en cours…";
      this.progress = 3;
      this.phase = "Démarrage";
      this.progressKind = "leaf";
      this.showProgress = true;
      this.busy = true;

      const fd = new FormData();
      fd.append("case_id", cid);
      if (this.model) fd.append("model", this.model);
      fd.append("pages", String(this.leafPages || 25));

      try {
        const r = await fetch("/api/leaf/stream", {
          method: "POST",
          body: fd,
        });
        if (!r.body) throw new Error("streaming non supporté");
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        const handle = (evt) => {
          if (!evt || typeof evt.type !== "string") return;
          if (evt.type === "progress") {
            const v = typeof evt.value === "number" ? evt.value : (evt.percent || 0) / 100;
            this.progress = Math.max(0, Math.min(100, Math.floor(v * 100)));
            if (this.progress >= 100) {
              this.phase = "Synthèse terminée ✅";
              this.$nextTick(() => {
                setTimeout(() => {
                  this.showProgress = false;
                }, 900);
              });
            }
          } else if (
            evt.type === "stage" ||
            evt.type === "status" ||
            evt.type === "log"
          ) {
            this.phase = evt.message || evt.label || "";
          } else if (evt.type === "error") {
            alert("Erreur LEAF: " + (evt.message || ""));
            this.$nextTick(() => {
              setTimeout(() => {
                this.showProgress = false;
              }, 1200);
            });
          } else if (evt.type === "end" || evt.type === "done") {
            this.progress = 100;
            this.phase = "Synthèse terminée ✅";
            try {
              if (evt.path) {
                const a = document.getElementById("leaf-download");
                if (a) {
                  a.href = evt.path;
                  a.style.display = "inline-block";
                }
              }
            } catch (_) {}
            this.$nextTick(() => {
              setTimeout(() => {
                this.showProgress = false;
              }, 900);
            });
          }
        };
        const parse = () => {
          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const raw = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (!raw) continue;
            if (raw.startsWith("data:")) {
              const js = raw.slice(5).trim();
              try {
                const evt = JSON.parse(js);
                handle(evt);
              } catch (_) {}
            }
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          parse();
        }
        parse();
      } catch (e) {
        alert("Erreur pendant la synthèse.");
      } finally {
        this.busy = false;
      }
    },

    // synthèse LONGUE (désactivée, remplacée par LEAF)
    async makeLongSummary() {
      alert(
        "La synthèse longue d’enquête a été désactivée.\n" +
        "Utilise la “Synthèse thématique (LEAF)” et/ou la “Synthèse chronologique” à la place."
      );
    },

    // ==========================
    // synthèse CHRONO (PV & rapports)
    // ==========================
    async makeChrono(mode = "long") {
      const cid = this.caseId;
      if (!cid) {
        alert("Aucune affaire sélectionnée");
        return;
      }

      this.task =
        mode === "short"
          ? "Synthèse chrono courte en cours…"
          : "Synthèse chronologique en cours…";
      this.progress = 3;
      this.phase = "Indexation des documents";
      this.progressKind = mode === "short" ? "chrono-short" : "chrono-long";
      this.showProgress = true;
      this.busy = true;

      // On masque les anciens liens de téléchargement au lancement
      try {
        const aLong = document.getElementById("chrono-long-download");
        if (aLong) {
          aLong.href = "#";
          aLong.style.display = "none";
        }
        const aShort = document.getElementById("chrono-short-download");
        if (aShort) {
          aShort.href = "#";
          aShort.style.display = "none";
        }
      } catch (_) {}

      const payload = {
        case_id: cid,
        case_title: "", // si tu veux forcer un titre, sinon laisser vide
        from_date: null,
        to_date: null,
        max_docs: null,
        mode: mode, // "long" ou "short"
        model: this.model || null,
      };

      try {
        const r = await fetch("/api/chrono/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!r.body) throw new Error("streaming non supporté");
        const reader = r.body.getReader();

        await parseSSEStream(reader, (evt) => {
          // evt = {stage: '...', ...} ou {path: '/storage/...'} en fin
          if (evt.stage) {
            this.phase = evt.message || evt.label || evt.stage;

            // Petites heuristiques pour une barre de progression approximative
            const stages = ["index", "doc_summary_progress", "short_report", "done"];
            const idx = stages.indexOf(evt.stage);
            if (idx >= 0) {
              this.progress = Math.max(this.progress, (idx + 1) * 20);
            }
            if (evt.total && typeof evt.done === "number") {
              // progression fine pendant la phase de résumés
              const pct = Math.min(95, Math.round((evt.done / evt.total) * 90));
              this.progress = Math.max(this.progress, pct);
            }
          }

          if (evt.path) {
            this.progress = 100;
            this.phase =
              mode === "short"
                ? "Synthèse chrono courte terminée ✅"
                : "Synthèse chronologique terminée ✅";

            // Met à jour les liens de téléchargement comme pour LEAF
            try {
              if (mode === "long") {
                const aLong = document.getElementById("chrono-long-download");
                if (aLong) {
                  aLong.href = evt.path;
                  aLong.style.display = "inline-block";
                }
              } else {
                const aShort = document.getElementById("chrono-short-download");
                if (aShort) {
                  aShort.href = evt.path;
                  aShort.style.display = "inline-block";
                }
              }
            } catch (_) {}

            this.$nextTick(() => {
              setTimeout(() => {
                this.showProgress = false;
              }, 900);
            });
          }
        });
      } catch (e) {
        alert("Erreur pendant la synthèse chrono.");
      } finally {
        this.busy = false;
      }
    },

    // === autosize textarea ===
    autoresize() {
      const ta = document.querySelector(".chat-input-bar textarea");
      if (!ta) return;
      ta.style.height = "auto";
      ta.style.height = ta.scrollHeight + "px";
    },
    setupAutoresize() {
      const ta = document.querySelector(".chat-input-bar textarea");
      if (!ta) return;
      ta.addEventListener("input", () => this.autoresize());
      // premier calcul
      this.autoresize();
    },

    // === helpers preview/téléchargement (encore utilisés pour d'autres cas éventuels) ===
    openChronoPreview(path) {
      try {
        window.open(path, "_blank");
      } catch (_) {
        console.log("Synthèse disponible:", path);
      }
    },
    offerDownload(path) {
      try {
        const a = document.createElement("a");
        a.href = path;
        const fname = path.split("/").pop() || "synthese_chrono.md";
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch (_) {}
    },
  };
}

// on expose pour Alpine
if (!window.chatApp) {
  window.chatApp = chatApp;
}
console.debug(
  "[axion] app.v2.js chargé, chatApp disponible :",
  typeof window.chatApp === "function"
);

// après avoir remplacé le message/loading par le rendu final (si tu as tempId)
if (typeof tempId !== "undefined" && tempId) {
  const mm = this.messages.find((x) => x.id === tempId);
  if (mm && mm.avatarHtml) {
    mm.avatarHtml = mm.avatarHtml
      .replace(/\btyping\b/g, "")
      .replace(/\bthinking\b/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }
  // Optionnel : mettre à jour le contenu final via updateAssistant si tu le veux
  // this.updateAssistant(tempId, this.renderAnswer(md), evtSources || []);
}
