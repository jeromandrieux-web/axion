(function(){
  // static/js/help.js
  // gère l'ouverture de la fenêtre d'aide.
  // Auteur: Jax

  "use strict";

  async function fetchHelpHtml(){
    try {
      const r = await fetch('/static/help.html');
      if(!r.ok) throw new Error('help file not found ('+r.status+')');
      return await r.text();
    } catch(e) {
      console.warn('[axion] fetchHelpHtml error', e);
      return `<div style="padding:1rem">Impossible de charger la page d'aide. Vérifie /static/help.html</div>`;
    }
  }

  function createModal(html){
    // remove existing if any
    const existing = document.getElementById('axion-help-modal');
    if(existing){ existing.remove(); }

    const wrapper = document.createElement('div');
    wrapper.id = 'axion-help-modal';
    wrapper.className = 'fixed inset-0 z-50 flex items-start justify-center p-4';
    wrapper.innerHTML = `
      <div class="absolute inset-0 bg-black/60" id="axion-help-overlay"></div>
      <div class="relative card p-5 w-[min(900px,98%)] max-h-[90vh] overflow-auto bg-[var(--bg)] text-[var(--text)]">
        <div class="flex items-center mb-3">
          <div class="font-semibold text-lg mr-3">Aide</div>
          <button id="axion-help-close" class="ml-auto pill text-xs">Fermer</button>
        </div>
        <div id="axion-help-content">${html}</div>
      </div>
    `;
    document.body.appendChild(wrapper);

    // close handlers
    document.getElementById('axion-help-overlay').addEventListener('click', closeModal);
    document.getElementById('axion-help-close').addEventListener('click', closeModal);
    function closeModal(){ const el = document.getElementById('axion-help-modal'); if(el) el.remove(); }
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    const btn = document.getElementById('help-btn');
    if(!btn) { console.debug('[axion] help-btn not found'); return; }

    btn.addEventListener('click', async (e)=>{
      e.preventDefault();
      btn.disabled = true;
      btn.textContent = 'Chargement…';
      const html = await fetchHelpHtml();
      createModal(html);
      btn.textContent = 'Aide';
      btn.disabled = false;
    });

    console.debug('[axion] help.js initialized');
  });

})();