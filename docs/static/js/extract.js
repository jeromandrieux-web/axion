// static/js/extract.js
// Axion - Extraction thématique et téléphones
// Auteur: Jax

(function () {
  "use strict";

  // ----------- Utils -----------
  function $(id) {
    return document.getElementById(id);
  }

  function setDisabled(el, disabled) {
    if (!el) return;
    el.disabled = !!disabled;
  }

  function showStatus(msg, isError = false) {
    const s = $("extractStatus");
    if (!s) return;
    s.innerHTML = isError ? `<span style="color:#b00020">${msg}</span>` : msg;
  }

  function clearOutput() {
    const out = $("extractOutput");
    if (out) out.textContent = "";
  }

  function appendOutput(text) {
    const out = $("extractOutput");
    if (!out) return;
    out.textContent += text;
    out.scrollTop = out.scrollHeight;
  }

  function safeJSONParse(s) {
    try {
      return JSON.parse(s);
    } catch {
      return null;
    }
  }

  // ----------- Alpine helper : récupérer l'instance du chat -----------
  function getChatApp() {
    try {
      const root = document.querySelector('[x-data="chatApp()"]');
      if (!root) return null;

      // Alpine v3 : __x.$data
      if (root.__x && root.__x.$data) {
        return root.__x.$data;
      }
      // fallback : _x_dataStack[0]
      if (root._x_dataStack && root._x_dataStack.length > 0) {
        const cmp = root._x_dataStack[root._x_dataStack.length - 1];
        return cmp.$data || cmp;
      }
    } catch (e) {
      console.warn("getChatApp failed", e);
    }
    return null;
  }

  // ----------- SSE helpers -----------
  async function _readSSEStream(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let currentEvent = "token";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trimEnd();
        buffer = buffer.slice(idx + 1);

        if (!line) continue;

        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim() || "message";
          continue;
        }
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          onEvent(currentEvent, data);
          continue;
        }

        onEvent(currentEvent, line);
      }
    }
  }

  async function streamFormPOST(url, formData, onEvent, onHTTPError) {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        accept: "text/event-stream",
      },
      body: formData,
    });

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      if (onHTTPError) onHTTPError(res.status, text);
      return;
    }

    await _readSSEStream(res, onEvent);
  }

  // ----------- Actions boutons "classiques" -----------
  async function runPhones() {
    const btnPhones = $("btnExtractPhones");
    const btnScan = $("btnExtractScan");
    const btnPerson = $("btnExtractPerson"); // peut ne pas exister, ce n'est pas grave

    const caseId =
      (
        $("case_id")?.value ||
        $("cases")?.value ||
        $("selectCase")?.value ||
        ""
      ).trim();

    let person = $("selectPerson")?.value?.trim() || "";
    const model =
      (
        $("model_name")?.value ||
        $("selectModel")?.value ||
        ""
      ).trim();

    clearOutput();
    showStatus("Extraction téléphones en cours…");

    if (!caseId) {
      showStatus("Veuillez sélectionner une affaire (case_id).", true);
      return;
    }
    if (!person) {
      person = "INCONNU";
    }

    setDisabled(btnPhones, true);
    setDisabled(btnScan, true);
    setDisabled(btnPerson, true);

    const fd = new FormData();
    fd.append("case_id", caseId);
    fd.append("person", person);
    fd.append("theme", "phones");
    fd.append("model", model || "deepseek-v3.1:671b-cloud");

    let strictValue = "0";
    try {
      const el = document.getElementById("extractStrict");
      if (el && el.checked) strictValue = "1";
    } catch (_) {}
    fd.append("strict_person", strictValue);

    // juste après avoir déterminé `caseId` et avant le lancement du stream :
    const app = getChatApp();
    if (app) {
      try { app.sending = true; app.showProgress = true; } catch (_) {}
    }

    await streamFormPOST(
      "/api/extract/scan/stream",
      fd,
      (eventType, dataRaw) => {
        if (eventType === "status") {
          showStatus(`📣 ${dataRaw}`);
          return;
        }
        if (eventType === "error") {
          showStatus(`Erreur: ${dataRaw}`, true);
          return;
        }
        if (eventType === "end") {
          showStatus("✅ Extraction des numéros terminée.");
          setDisabled(btnPhones, false);
          setDisabled(btnScan, false);
          setDisabled(btnPerson, false);
          return;
        }

        const payload = dataRaw.startsWith("data:")
          ? dataRaw.slice(5).trim()
          : dataRaw;
        const j = safeJSONParse(payload);
        if (j && j.content) {
          appendOutput(j.content);
        } else {
          appendOutput(payload);
        }
      },
      (httpStatus, text) => {
        showStatus(`Requête échouée (${httpStatus}). ${text || ""}`, true);
        setDisabled(btnPhones, false);
        setDisabled(btnScan, false);
        setDisabled(btnPerson, false);
      }
    );

    // dans tous les chemins de sortie (avant chaque return / dans finally) :
    if (app) {
      try { app.sending = false; app.showProgress = false; } catch (_) {}
    }

    setDisabled(btnPhones, false);
    setDisabled(btnScan, false);
    setDisabled(btnPerson, false);
  }

  async function runScan() {
    const btnPhones = $("btnExtractPhones");
    const btnScan = $("btnExtractScan");
    const btnPerson = $("btnExtractPerson");

    const caseId =
      (
        $("case_id")?.value ||
        $("cases")?.value ||
        $("selectCase")?.value ||
        ""
      ).trim();

    let person = $("selectPerson")?.value?.trim() || "";
    const model =
      (
        $("model_name")?.value ||
        $("selectModel")?.value ||
        ""
      ).trim();
    const theme = $("selectTheme")?.value?.trim() || "";

    clearOutput();
    showStatus("Scan thématique en cours…");

    if (!caseId) {
      showStatus("Veuillez sélectionner une affaire (case_id).", true);
      return;
    }
    if (!theme) {
      showStatus("Veuillez sélectionner un thème.", true);
      return;
    }
    if (!person) {
      person = "INCONNU";
    }

    setDisabled(btnPhones, true);
    setDisabled(btnScan, true);
    setDisabled(btnPerson, true);

    const fd = new FormData();
    fd.append("case_id", caseId);
    fd.append("person", person);
    fd.append("theme", theme);
    fd.append("model", model || "deepseek-v3.1:671b-cloud");

    let strictValue = "0";
    try {
      const el = document.getElementById("extractStrict");
      if (el && el.checked) strictValue = "1";
    } catch (_) {}
    fd.append("strict_person", strictValue);

    // juste après avoir déterminé `caseId` et avant le lancement du stream :
    const app = getChatApp();
    if (app) {
      try { app.sending = true; app.showProgress = true; } catch (_) {}
    }

    await streamFormPOST(
      "/api/extract/scan/stream",
      fd,
      (eventType, dataRaw) => {
        if (eventType === "status") {
          showStatus(`📣 ${dataRaw}`);
          return;
        }
        if (eventType === "error") {
          showStatus(`Erreur: ${dataRaw}`, true);
          return;
        }
        if (eventType === "end") {
          showStatus("✅ Scan thématique terminé.");
          setDisabled(btnPhones, false);
          setDisabled(btnScan, false);
          setDisabled(btnPerson, false);
          return;
        }

        const payload = dataRaw.startsWith("data:")
          ? dataRaw.slice(5).trim()
          : dataRaw;
        const j = safeJSONParse(payload);
        if (j && (j.hit || j.doc || j.excerpt)) {
          const title = j.doc?.title || j.doc?.name || "Document";
          const loc = j.doc?.path || j.doc?.id || "";
          const line = j.excerpt || j.hit || "";
          appendOutput(`\n• ${title}${loc ? ` — ${loc}` : ""}\n  ${line}`);
        } else {
          appendOutput(payload);
        }
      },
      (httpStatus, text) => {
        showStatus(`Requête échouée (${httpStatus}). ${text || ""}`, true);
        setDisabled(btnPhones, false);
        setDisabled(btnScan, false);
        setDisabled(btnPerson, false);
      }
    );

    // dans tous les chemins de sortie (avant chaque return / dans finally) :
    if (app) {
      try { app.sending = false; app.showProgress = false; } catch (_) {}
    }

    setDisabled(btnPhones, false);
    setDisabled(btnScan, false);
    setDisabled(btnPerson, false);
  }

  // ----------- Bind UI -----------
  function bind() {
    const b1 = $("btnExtractPhones");
    const b2 = $("btnExtractScan");
    if (b1) b1.addEventListener("click", runPhones);
    if (b2) b2.addEventListener("click", runScan);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }

  // ----------- API réutilisable (thématique + fiche personne) -----------
  window.axionExtract = window.axionExtract || {
    // extraction thématique réutilisable
    extractRun: async function (
      { caseId, theme, person, model, strict_person } = {},
      onEvent = () => {}
    ) {
      if (!caseId) throw new Error("case_id manquant");
      const url = "/api/extract/scan/stream";

      const fd = new FormData();
      fd.append("case_id", String(caseId));
      fd.append("person", person || "INCONNU");
      fd.append("theme", (theme || "phones").toLowerCase());
      fd.append("model", model || "deepseek-v3.1:671b-cloud");
      fd.append("strict_person", strict_person ? "1" : "0");

      const resp = await fetch(url, {
        method: "POST",
        headers: { accept: "text/event-stream" },
        body: fd,
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(() => "<no body>");
        throw new Error("HTTP " + resp.status + " " + txt);
      }

      if (resp.body) {
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const block = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (block.startsWith("data:")) {
              try {
                const ev = JSON.parse(block.slice(5).trim());
                onEvent(ev);
              } catch (_) {
                onEvent({ type: "raw", data: block });
              }
            } else {
              onEvent({ type: "raw", data: block });
            }
          }
        }
      }
      return { ok: true, streamed: true };
    },

    // API fiche personne
    extractPersonRun: async function (
      { caseId, person, model, aliases } = {},
      onEvent = () => {}
    ) {
      if (!caseId) throw new Error("case_id manquant");
      if (!person) throw new Error("person manquant");

      const fd = new FormData();
      fd.append("case_id", String(caseId));
      fd.append("person", person);
      fd.append("model", model || "deepseek-v3.1:671b-cloud");
      if (aliases) {
        fd.append(
          "aliases",
          Array.isArray(aliases) ? aliases.join(" | ") : String(aliases)
        );
      }

      const resp = await fetch("/api/extract/person/profile", {
        method: "POST",
        body: fd,
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        throw new Error("HTTP " + resp.status + " " + txt);
      }

      const data = await resp.json().catch(() => null);
      if (!data) {
        throw new Error(
          "Réponse JSON invalide depuis /api/extract/person/profile"
        );
      }

      try {
        onEvent({ type: "done", data });
      } catch (_) {}

      return data;
    },
  };

  // ---------------------------------------------------------------------
  // 📞 Fiche téléphone — backend + intégration dans le chat
  // ---------------------------------------------------------------------

  // 1) Appel backend, renvoie les données JSON
  async function extractPhoneRun(phone, caseId) {
    if (!caseId) {
      throw new Error("Aucune affaire sélectionnée pour la fiche téléphone.");
    }

    const fd = new FormData();
    fd.append("case_id", caseId || "");
    fd.append("number", phone);

    const resp = await fetch("/api/phones/report", {
      method: "POST",
      body: fd,
    });

    let data = null;
    try {
      data = await resp.json();
    } catch (_) {
      // on laisse data = null
    }

    if (!resp.ok) {
      console.error("phones/report error", resp.status, data);
      const msg =
        (data && (data.detail || data.message)) ||
        "Erreur lors de la génération de la fiche téléphone.";
      throw new Error(msg);
    }

    if (!data || !data.path) {
      throw new Error("Réponse inattendue du serveur (pas de chemin de fichier).");
    }

    return data; // { case_id, input_number, normalized_number, hits_count, path }
  }

  // 2) Fonction globale appelée depuis index.html
  window.sendPhoneReport = function (phoneRaw, explicitCaseId) {
    const phone = (phoneRaw || "").trim();
    if (!phone) {
      alert("Indique un numéro de téléphone");
      // pour que le .finally de l'Alpine ne plante pas
      return Promise.resolve();
    }

    // On récupère l'instance du chat pour l'intégration visuelle
    const app = getChatApp();

    // caseId : priorité au paramètre explicite, sinon state Alpine
    const caseId =
      (explicitCaseId || "").trim() ||
      (app && app.caseId ? String(app.caseId).trim() : "");

    if (!caseId) {
      alert("Sélectionne d’abord une affaire pour générer une fiche téléphone.");
      return Promise.resolve();
    }

    // Si pas d'app (Alpine pas dispo pour une raison quelconque),
    // on fait au moins le backend + lien de téléchargement.
    if (!app) {
      return extractPhoneRun(phone, caseId)
        .then((data) => {
          // mise à jour du lien de téléchargement
          const a = document.getElementById("phone-download");
          if (a) {
            a.href = data.path;
            a.style.display = "inline-block";
          }
          alert(
            `Fiche téléphone générée pour le numéro ${data.normalized_number} (${data.hits_count} occurrence(s)).`
          );
        })
        .catch((err) => {
          alert(err.message || "Erreur lors de la génération de la fiche téléphone.");
        });
    }

    // --------------------------
    // 💬 Intégration dans le chat
    // --------------------------

    // 1) bulle utilisateur
    if (typeof app.pushUser === "function") {
      app.pushUser(`Fiche téléphone demandée pour le numéro : ${phone}`);
    }

    // 2) bulle assistant "en cours..."
    let aId = null;
    if (typeof app.pushAssistantPlaceholder === "function") {
      aId = app.pushAssistantPlaceholder();
    }

    // 3) appel backend
    return extractPhoneRun(phone, caseId)
      .then((data) => {
        // maj lien de téléchargement
        try {
          const a = document.getElementById("phone-download");
          if (a) {
            a.href = data.path;
            a.style.display = "inline-block";
          }
        } catch (_) {}

        // texte récapitulatif pour la bulle assistant
        let txt;
        if (!data.hits_count) {
          txt =
            `Aucune occurrence du numéro **${data.normalized_number}** ` +
            `n’a été trouvée dans l’affaire \`${data.case_id}\`.\n\n` +
            `La fiche téléphone a tout de même été générée (structure standard) ` +
            `et reste téléchargeable via le lien sous la boîte à outils.`;
        } else {
          txt =
            `Fiche téléphone générée pour le numéro **${data.normalized_number}** ` +
            `dans l’affaire \`${data.case_id}\`.\n\n` +
            `- Occurrences détectées : **${data.hits_count}**\n` +
            `- Le fichier détaillé (contexte par document et extraits) est téléchargeable ` +
            `via le lien « 📥 Télécharger la fiche téléphone » sous la boîte à outils.`;
        }

        if (aId && typeof app.updateAssistant === "function") {
          const html = app.renderAnswer ? app.renderAnswer(txt) : txt;
          app.updateAssistant(aId, html);
        }
      })
      .catch((err) => {
        console.error("sendPhoneReport error", err);
        if (aId && typeof app.updateAssistant === "function") {
          const msg =
            err.message ||
            "Erreur lors de la génération de la fiche téléphone.";
          const html = app.escape
            ? '<span class="text-red-400">' + app.escape(msg) + "</span>"
            : '<span class="text-red-400">' + msg + "</span>";
          app.updateAssistant(aId, html);
        } else {
          alert(err.message || "Erreur lors de la génération de la fiche téléphone.");
        }
      });
  };
})();
