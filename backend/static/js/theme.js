// /static/js/theme.js
// Thème synchronisé avec la machine + possibilité de forcer light/dark
// Auteur: Jax

(function () {
  "use strict";

  console.log("[theme.js] sync système + override possible");

  const OVERRIDE_KEY = "theme_override"; // "light" | "dark" | null
  const DEFAULT_THEME = "dark";

  // ---- Utils stockage override ----
  function getOverride() {
    try {
      const v = localStorage.getItem(OVERRIDE_KEY);
      if (v === "light" || v === "dark") return v;
    } catch (_) {}
    return null;
  }

  function setOverride(modeOrNull) {
    try {
      if (modeOrNull === null) {
        localStorage.removeItem(OVERRIDE_KEY);
      } else if (modeOrNull === "light" || modeOrNull === "dark") {
        localStorage.setItem(OVERRIDE_KEY, modeOrNull);
      }
    } catch (_) {}
  }

  // ---- Thème système ----
  function getSystemTheme() {
    try {
      if (window.matchMedia) {
        const mqDark = window.matchMedia("(prefers-color-scheme: dark)");
        if (mqDark.matches) return "dark";
        return "light";
      }
    } catch (e) {
      console.warn("[theme.js] matchMedia error", e);
    }
    // fallback : si pas de matchMedia, on part sur dark par défaut
    return DEFAULT_THEME;
  }

  // Thème effectif = override si présent, sinon système
  function getEffectiveTheme() {
    const o = getOverride();
    if (o) return o;
    return getSystemTheme();
  }

  // ---- Application visuelle ----
  function applyTheme(mode) {
    if (!mode) mode = DEFAULT_THEME;

    document.documentElement.dataset.theme = mode;
    document.body.dataset.theme = mode;
    document.documentElement.style.colorScheme = (mode === "light" ? "light" : "dark");

    document.body.classList.toggle("is-light", mode === "light");
    document.body.classList.toggle("is-dark", mode === "dark");

    const icon = document.getElementById("theme-icon");
    if (icon) {
      icon.textContent = (mode === "light" ? "🌙" : "☀️");
    }

    try {
      document.dispatchEvent(
        new CustomEvent("axion-theme-changed", { detail: { theme: mode } })
      );
    } catch (e) {
      console.warn("[theme.js] CustomEvent error (non bloquant)", e);
    }
  }

  // ---- Listener sur les changements de thème système ----
  function setupSystemListener() {
    if (!window.matchMedia) return;
    try {
      const mqDark = window.matchMedia("(prefers-color-scheme: dark)");

      const handler = function (e) {
        // si override, on ne suit pas le système
        if (getOverride()) return;

        const mode = e.matches ? "dark" : "light";
        console.log("[theme.js] système changé ->", mode);
        applyTheme(mode);
      };

      if (typeof mqDark.addEventListener === "function") {
        mqDark.addEventListener("change", handler);
      } else if (typeof mqDark.addListener === "function") {
        mqDark.addListener(handler); // compat vieux Safari
      }
    } catch (e) {
      console.warn("[theme.js] setupSystemListener error", e);
    }
  }

  // ---- Init global ----
  function init() {
    console.log("[theme.js] init");
    const effective = getEffectiveTheme();
    applyTheme(effective);
    setupSystemListener();

    // si tu n'as PAS mis d'onclick dans le HTML, on câble ici
    const btn = document.getElementById("theme-toggle");
    if (btn && !btn.getAttribute("onclick")) {
      btn.addEventListener("click", function () {
        window.axionTheme.toggle();
      });
    }
  }

  // ---- API globale ----
  window.axionTheme = {
    // thème effectif (ce qui est appliqué visuellement)
    getEffective: getEffectiveTheme,
    // thème système brut (utile si besoin)
    getSystem: getSystemTheme,
    // FORCER clair/sombre (override)
    setOverride: function (mode) {
      if (mode !== "light" && mode !== "dark") return;
      setOverride(mode);
      applyTheme(mode);
    },
    // Toggle = on bascule entre light/dark et on force
    toggle: function () {
      const current = getEffectiveTheme();
      const next = current === "light" ? "dark" : "light";
      setOverride(next);
      applyTheme(next);
    },
    // Revenir à "suivre la machine"
    useSystem: function () {
      setOverride(null);
      const sys = getSystemTheme();
      applyTheme(sys);
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
