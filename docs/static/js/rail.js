// /static/js/rail.js
// gère l'affichage/masquage du rail gauche (app-rail)
// Auteur: Jax

(function () {
  "use strict";

  const STORAGE_KEY = "rail-state"; // "shown" | "hidden"
  const DEFAULT_VISIBLE = true;

  function applyRail(visible, opts) {
    opts = opts || {};
    const shouldPersist = opts.persist !== false;

    document.body.classList.toggle("rail-collapsed", !visible);

    if (shouldPersist) {
      try {
        localStorage.setItem(STORAGE_KEY, visible ? "shown" : "hidden");
      } catch (_) {}
    }

    const icon = document.getElementById("rail-icon");
    if (icon) {
      icon.textContent = visible ? "⏴" : "⏵";
    }

    document.dispatchEvent(
      new CustomEvent("axion-rail-changed", { detail: { visible } })
    );
  }

  function getInitialState() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved === "shown") return true;
      if (saved === "hidden") return false;
    } catch (_) {}
    return DEFAULT_VISIBLE;
  }

  window.axionRail = {
    apply: applyRail,
    toggle: function () {
      applyRail(!getInitialState());
    },
    show: () => applyRail(true),
    hide: () => applyRail(false),
  };

  document.addEventListener("DOMContentLoaded", function () {
    applyRail(getInitialState(), { persist: false });

    const btn = document.getElementById("rail-toggle");
    if (btn) {
      btn.addEventListener("click", () => window.axionRail.toggle());
    }
  });
})();
