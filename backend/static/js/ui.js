// static/js/ui.js
// Axion - Gestion de l'interface utilisateur
// Auteur: Jax

(function(){
  "use strict";
  document.addEventListener('DOMContentLoaded', () => {
    const bg = document.getElementById('bgLogo');
    if (!bg) return;
    const ta = document.querySelector('textarea');
    if (ta) {
      ta.addEventListener('input', () => bg.classList.add('is-hidden'), { once: true });
      ta.addEventListener('focus', () => bg.classList.add('is-hidden'), { once: true });

      // --- Ajout : autosize du textarea ---
      // remplace la sélection unique par une initialisation plus robuste
      const findTextareas = () => Array.from(document.querySelectorAll('.chat-input-bar textarea'));

      const initAutosizeFor = (el) => {
        if (el._autosizeInitialized) return;
        el._autosizeInitialized = true;

        // cacher logo au premier input/focus
        const hideLogoOnce = () => bg && bg.classList && bg.classList.add('is-hidden');
        el.addEventListener('input', hideLogoOnce, { once: true });
        el.addEventListener('focus', hideLogoOnce, { once: true });

        const autosize = (node) => {
          node.style.height = 'auto';
          const computed = getComputedStyle(node);
          const maxHeight = parseInt(computed.maxHeight, 10) || 0;
          const target = node.scrollHeight;
          if (maxHeight && target > maxHeight) {
            node.style.height = maxHeight + 'px';
            node.style.overflowY = 'auto';
          } else {
            node.style.height = target + 'px';
            node.style.overflowY = 'hidden';
          }
        };

        const schedule = () => requestAnimationFrame(() => autosize(el));
        el.addEventListener('input', schedule);
        el.addEventListener('paste', schedule);
        el.addEventListener('cut', schedule);
        el.addEventListener('change', schedule);
        el.addEventListener('keydown', schedule);
        // init si déjà du texte
        schedule();
      };

      const initAllTextareas = () => {
        findTextareas().forEach(initAutosizeFor);
      };

      // initialisation immédiate (si présent)
      initAllTextareas();

      // observer pour textarea ajoutés dynamiquement (Alpine)
      const mo = new MutationObserver((mutations) => {
        if (mutations.some(m => m.addedNodes && m.addedNodes.length)) {
          initAllTextareas();
        }
      });
      mo.observe(document.body, { childList: true, subtree: true });
      // --- fin autosize ---
    }
    const chatRoot = document.querySelector('main') || document.body;
    const hideIfMessages = () => {
      const hasMsgs = document.querySelectorAll('.chat-bubble').length > 0;
      if (hasMsgs) bg.classList.add('is-hidden');
    };
    new MutationObserver(hideIfMessages).observe(chatRoot, { childList: true, subtree: true });
    hideIfMessages();

    window.addEventListener && window.addEventListener("alpine:init", ()=> console.debug("[axion] alpine:init"));
    window.addEventListener && window.addEventListener("DOMContentLoaded", ()=> console.debug("[axion] DOMContentLoaded (ui.js)"));
    console.debug('[axion] ui.js loaded');

    // NOTE: global Enter handler removed to avoid duplicate send calls.
  });
})();