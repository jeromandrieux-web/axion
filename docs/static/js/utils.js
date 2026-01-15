// static/js/utils.js
// Axion - Utilitaires JavaScript divers
// Auteur: Jax

(function(){
  "use strict";
  function escapeHtml(s){
    if(s==null) return "";
    return String(s).replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]));
  }
  function nl2br(s){ return String(s||'').replaceAll('\n','<br>'); }
  function formatEta(s){
    if(!s) return '';
    const m = Math.floor(s/60), sec = Math.max(0, Math.round(s%60));
    if(m>0) return `${m}min ${sec}s`; return `${sec}s`;
  }
  function firstArrayProp(obj){
    if(!obj || typeof obj !== 'object') return null;
    for(const k of Object.keys(obj)){
      if(Array.isArray(obj[k])) return obj[k];
    }
    return null;
  }

  window.axionUtils = {
    escapeHtml, nl2br, formatEta, firstArrayProp
  };

  // Exposer formatEta globalement pour les expressions Alpine qui appellent formatEta(...)
  // (évite ReferenceError si le template utilise formatEta directement)
  window.formatEta = formatEta;

  console.debug('[axion] utils.js loaded');
})();