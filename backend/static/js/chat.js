// static/js/chat.js
// Fichier legacy allégé : la logique de chat est désormais dans app.v2.js (chatApp).
// On garde uniquement un petit pont de compatibilité éventuel.

(function () {
  "use strict";

  console.debug("[axion] chat.js (legacy stub) chargé");

  // Pont de compatibilité : certaines anciennes pages/appels peuvent encore tenter
  // d'utiliser window.chatAppFactory(). On le redirige vers chatApp() si présent.
  if (!window.chatAppFactory) {
    window.chatAppFactory = function () {
      if (typeof window.chatApp === "function") {
        console.debug("[axion] chatAppFactory appelé → redirection vers chatApp()");
        return window.chatApp();
      }
      console.warn(
        "[axion] chatAppFactory appelé mais window.chatApp n'est pas défini. " +
          "Assure-toi que app.v2.js est bien chargé avant ce script."
      );
      return {};
    };
  }
})();
