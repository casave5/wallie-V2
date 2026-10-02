# Casavita VTuber — dónde lo dejamos (2026-09-29)

## ✅ YA FUNCIONA
- **Voz**: Casavita habla con Piper `daniela-high` + GPU NVIDIA + filtro anime.
  Probado en el dashboard: `http://127.0.0.1:8765` → Personality → Test chat reply.
- **Cerebro**: Groq `openai/gpt-oss-120b` (rápido, gratis). Perfil: `profiles/casavita.yaml`.
- **Bot de Telegram**: `@CasaveBot` funcionando (notas de voz + texto).
- Personalidad: no dice su origen, texto plano, sin emojis en voz.

## ❌ LO QUE FALLA / PENDIENTE
- **Oído en vivo**: Wallie solo escucha en Windows. En Linux hay que reescribir
  `hearing/capture.py`. Tu micro además da poca señal. **Decidido: dejarlo para después.**
- **Avatar**: falta instalar/configurar VTube Studio y elegir un modelo Live2D.
- **Chat de Twitch**: falta conectar el chat de Wallie a tu canal (necesita tu nombre de canal).

## 🎯 SIGUIENTE PASO (opción C elegida)
1. Conocer el **nombre de tu canal de Twitch**.
2. Decidir el **avatar**: ¿ya tienes un modelo Live2D (`.model3.json`) o buscamos uno?
3. Instalar **VTube Studio** y añadirlo a Wallie.
4. Conectar el **chat de Twitch** para que Casavita lo lea y comente.
5. Probar en vivo con OBS.

## 🔑 PARA CONTINUAR
- Abrazo: abre opencode en la laptop y di "seguimos con Casavita VTuber".
- Todo el contexto está en la memoria + este archivo.
- Para hablar con Casavita estando lejos: mándale una **nota de voz** a `@CasaveBot` en Telegram.
  (Ojo: el bot de Telegram NO sabe nada del proyecto Wallie, solo conversa contigo.)
