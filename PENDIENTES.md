# Casavita VTuber — dónde lo dejamos (2026-10-02)

> Esto manda sobre cualquier nota anterior. Lo de "solo funciona en Windows",
> "falta el oído" y "falta el avatar" ya NO es cierto.

## ✅ YA FUNCIONA (verificado en vivo)

- **Oído en Linux** (lo que faltaba de verdad): `hearing/capture.py` reescrito.
  - El micro es un USB Razer BlackShark V2 X (`hw:3,0`) y **solo admite un
    consumidor**: `arecord` reintenta 2 veces y si no, cae a `parec`.
  - PipeWire retiene `hw:3,0`, por eso hace falta ese plan B.
  - No se puede comprobar señal al arrancar (el micro da ceros digitales en
    silencio real): decide por "backend arrancó", no por nivel de audio.
  - Watchdog: avisa tras 30 s sin señal.
- **Dos modelos de whisper**: `base` transcribe, `tiny` solo busca el nombre.
  `end_silence_sec: 1.3` (antes partía la palabra clave).
  NO optimizar el audio por trozos: parte palabras y las respuestas salen
  incoherentes (probado y revertido).
- **Formas de hablarle**:
  - `Ctrl+Shift+Space` alterna el oído (funciona con y sin decir "Casavita").
    Se cierra solo a los 8 s.
  - Decir **"Casavita"** al principio de la frase también la abre.
- **Avatar**: motor Live2D web propio (`avatar/web/`, node) servido en
  `http://127.0.0.1:8100`, dentro de una ventana Qt6 + QtWebEngine
  (`~/casavita-escritorio.sh`). Se mueve con clic derecho en su botón de la
  barra de tareas → "Mover". **Va en la segunda pantalla, que es la que se
  comparte en el stream, y el audio de ella ya entra por OBS.** La boca se
  mueve al hablar (verificado).
- **Voz**: Piper `es_AR-daniela-high` + GPU (`piper_use_cuda: true`) con
  cadena de filtro anime (`piper_postprocess`, length_scale 1.15, atempo 0.80).
- **Cerebro**: Groq `openai/gpt-oss-120b`. `vision.enabled: false` (no ve la
  pantalla) y `spontaneous_monologue: false`: **solo habla si le hablas tú, si
  alguien escribe en el chat o en el chequeo de silencio**, nunca por su cuenta.
- **Personalidad** (arreglada 2026-10-02):
  - `reveal_ai: false` + reglas para no admitir que es IA (puede hablar de IA
    como tema, pero no de sí misma). Validado: se lo esquiva.
  - `chat.banter` **ahora sí manda**: con `false` desaparecen las reglas de
    guasa del prompt (antes estaban fijas y el interruptor no hacía nada).
  - Las notas del dueño guardan lo que dice **casave**, no lo que dice Casavita
    (el chequeo de silencio citaba su propia voz).
- **Bot de Telegram**: `@CasaveBot` (notas de voz y texto). Ojo: no sabe nada
  del proyecto Wallie, solo conversa.

## ❌ LO QUE FALTA

1. **Chat de Twitch de verdad.** El perfil tiene `twitch_enabled: true` pero
   nunca se ha conectado: falta el **nombre del canal** y el **token OAuth**.
   Mientras tanto no lee el chat de tu canal.
2. **Prueba en vivo** con gente: ella + chat + stream a la vez.
3. (Opcional) Encender `vision` para que vea la pantalla y `spontaneous_monologue`
   para que hable sola. Ahora mismo está todo apagado a propósito.

## 🧪 TESTS

```bash
cd ~/wallie-V2 && .venv/bin/python -m pytest tests -q     # 46 tests
```

## 🔑 PARA CONTINUAR

- **Arrancar/parar el avatar**:
  - `~/casavita-escritorio.sh` → abrir · `--alto-avatar 900` → más alta
  - `~/casavita-escritorio.sh --capa` → a pantalla completa
  - `~/casavita-escritorio.sh --stop` → cerrar
- **Servicio**: `systemctl --user restart wallie.service` (los logs van con
  `journalctl --user -u wallie -f`).
- **Dashboard**: `http://127.0.0.1:8765` → Personality → Test chat reply.
- **Código**: commit `e889b08`, ya subido a la fork `TU_USUARIO/wallie-V2`.
  - `origin` = repo del autor (solo para traerse novedades).
  - `fork` = el tuyo → `git push fork main`.
- **LECCIÓN**: el HEAD del repo estaba muy desfasado. **No restaurar con git**:
  se perdería trabajo local. Leer la zona exacta y validar con `ast.parse`.
- Para hablar con Casavita estando lejos: nota de voz a `@CasaveBot` en Telegram.