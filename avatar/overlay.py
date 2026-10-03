#!/usr/bin/env python3
"""Casavita en el escritorio (KDE Plasma / Wayland).

Ventana normal con barra de titulo, del tamano del avatar y con fondo
transparente, mostrando el avatar que sirve /avatar/web. Se mueve arrastrando
por la barra de titulo, como cualquier ventana. No toca OBS.

    ~/casavita-escritorio.sh                  -> abrir (640x800)
    ~/casavita-escritorio.sh --alto-avatar 900 -> mas alta
    ~/casavita-escritorio.sh --capa           -> sin barra, a pantalla completa
    ~/casavita-escritorio.sh --stop           -> cerrar

Notas de por que va con barra de titulo:
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView

URL_BASE = "http://127.0.0.1:8100/"
LOG = "/tmp/opencode/overlay.log"
POSICION = os.path.expanduser("~/.config/casavita/posicion.json")
ICONO_BARRA = os.path.expanduser(
    "~/.local/share/icons/hicolor/256x256/apps/casavita-barra.png")

# Cada cuanto se recuerda que la ventana va por delante. KWin la reapila al
# maximizar otra ventana, y re-afirmarlo nosotros es mas fiable que pelearnos
# con las reglas de apilado del compositor.
MANTENER_ENCIMA_MS = 1200


def registro(mensaje: str) -> None:
    """Log en fichero: permite diagnosticar sin depender de ver la pantalla."""
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{mensaje}\n")
    except OSError:
        pass


def leer_posicion() -> dict | None:
    try:
        with open(POSICION, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def guardar_posicion(x: int, y: int, modo: str = "", ancho: int = 0,
                     alto: int = 0) -> None:
    """Recuerda donde y de que tamaño queda la ventana entre sesiones."""
    try:
        os.makedirs(os.path.dirname(POSICION), exist_ok=True)
        with open(POSICION, "w", encoding="utf-8") as fh:
            json.dump({"x": int(x), "y": int(y), "modo": modo,
                       "ancho": int(ancho), "alto": int(alto)}, fh)
    except OSError:
        pass


class VentanaAvatar(QWebEngineView):
    """Ventana normal con el avatar dentro, ajustado a su tamano."""

    def __init__(self, url: str, raton: bool, modo: str,
                 alto_avatar: int = 800, sin_barra: bool = False,
                 ancho_inicial: int = 0) -> None:
        super().__init__()
        flags = Qt.WindowType.WindowStaysOnTopHint   # siempre delante
        if sin_barra:
            flags |= Qt.WindowType.FramelessWindowHint
        self.setWindowFlags(flags)
        self.setWindowTitle("Casavita")

        # Icono propio en la barra de tareas: KWin usa el del Widget, no el del
        # .desktop del lanzador. Y DesktopFileName la agrupa con ese lanzador.
        icono = QIcon(ICONO_BARRA)
        if not icono.isNull():
            self.setWindowIcon(icono)

        # Las dos cosas juntas hacen que WebEngine deje de pintar blanco:
        # fondo translucido en el widget + fondo transparente en la pagina.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))
        self.setStyleSheet("QWebEngineView { background: transparent; }")

        self._modo = modo
        self.alto_avatar = alto_avatar
        if not raton:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # El tamaño se fija ANTES de pedir la pagina: si no, el lienzo arranca
        # con el tamano por defecto de Qt, la muñequita se dibuja a esa escala
        # y luego da un salto al ajustarse a la ventana.
        if ancho_inicial:
            self.resize(ancho_inicial, alto_avatar)

        self._atajos()
        self._instalar_filtro_rueda()
        self.loadFinished.connect(self._al_cargar)
        self.setUrl(QUrl(url))          # <- sin esto la ventana sale vacia
        self._timer_pos = QTimer(self)
        self._timer_pos.timeout.connect(self._guardar_pos)
        self._timer_pos.start(2000)          # guarda la posicion cada 2 s
        self._timer_encima = QTimer(self)
        self._timer_encima.timeout.connect(self._mantener_encima)
        self._timer_encima.start(MANTENER_ENCIMA_MS)

        registro(f"creada titulo={self.windowTitle()!r} sin_barra={sin_barra} raton={raton}")

    # --- tamaño de la ventana ---------------------------------------------
    def _cambiar_tamano(self, delta_alto: int) -> None:
        """Agranda o encoge la ventana manteniendo la proporción del avatar.

        Sin barra ni bordes no hay dónde agarrar para redimensionar, así que
        el tamaño se ajusta con Ctrl+Arriba / Ctrl+Abajo (o Ctrl+rueda). El
        alto queda limitado para que siga siendo usable.
        """
        proporcion = 0.8            # natural 1600x2000
        alto = max(200, min(1600, self.height() + delta_alto))
        ancho = max(160, int(round(alto * proporcion)))
        self.resize(ancho, alto)
        guardar_posicion(self.x(), self.y(), self._modo, ancho, alto)

    def _atajos(self) -> None:
        """Atajos con QShortcut.

        No sirven los keyPressEvent de la ventana: Qt WebEngine se queda con
        las teclas para mandarlas a la pagina, asi que nunca llegaban.
        Con ApplicationShortcut funcionan con la ventana activa sin necesidad
        de que tenga el foco de teclado.
        """
        self._atajos_creados = []
        self._filtros_rueda = set()
        for secuencia, delta in (("Ctrl+Down", -40), ("Ctrl+Up", 40)):
            atajo = QShortcut(QKeySequence(secuencia), self)
            atajo.setContext(Qt.ShortcutContext.ApplicationShortcut)
            atajo.activated.connect(lambda d=delta: self._cambiar_tamano(d))
            self._atajos_creados.append(atajo)
        salir = QShortcut(QKeySequence("Esc"), self)
        salir.setContext(Qt.ShortcutContext.ApplicationShortcut)
        salir.activated.connect(self.close)
        self._atajos_creados.append(salir)

    def _instalar_filtro_rueda(self) -> None:
        """Pone el filtro tambien en los widgets internos de WebEngine.

        La rueda no la recibe la ventana: la recibe un widget hijo interno, de
        modo que con filtrar solo la vista no llegaba nada.
        """
        from PyQt6.QtWidgets import QWidget
        self.installEventFilter(self)
        for hijo in self.findChildren(QWidget):
            self._filtros_rueda.add(hijo)
            hijo.installEventFilter(self)
        registro(f"filtro de rueda en {len(self._filtros_rueda) + 1} widgets")

    def eventFilter(self, obj, event) -> bool:
        """Ctrl + rueda sobre la ventana cambia el tamaño."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QWheelEvent
        if (isinstance(event, QWheelEvent)
                and event.type() == QEvent.Type.Wheel
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._cambiar_tamano(-40 if event.angleDelta().y() > 0 else 40)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    # --- ciclo de vida ------------------------------------------------------
    def _al_cargar(self, ok: bool) -> None:
        self.setWindowTitle("Casavita")
        self._mantener_encima()
        if self._modo == "capa":
            self._ceñir(self.alto_avatar)
        registro(f"cargada ok={ok} titulo={self.windowTitle()!r}")
        if ok:
            self._instalar_filtro_rueda()
            QTimer.singleShot(1500, self._instalar_filtro_rueda)
            self._programar_sonda()      # medir con la pagina ya lista
        # Sondeo de diagnostico: que ve la pagina realmente.
        QTimer.singleShot(1800, lambda: self.page().runJavaScript(
            "window.__casavita"
            " ? 'canvas ' + __casavita.canvas().ancho + 'x' + __casavita.canvas().alto"
            "   + ' ventana ' + innerWidth + 'x' + innerHeight"
            "   + ' natural ' + __casavita.natural().ancho + 'x' + __casavita.natural().alto"
            " : 'SIN PUENTE'",
            lambda v: registro(f"sondeo: {v}"),
        ))

    def _mantener_encima(self) -> None:
        if self.isVisible():
            self.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        guardar_posicion(self.x(), self.y(), self._modo,
                         self.width(), self.height())
        super().closeEvent(event)

    def _guardar_pos(self) -> None:
        guardar_posicion(self.x(), self.y(), self._modo, self.width(), self.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        """Registra cada cambio de tamaño o posición.

        Así se puede consultar el estado real de la ventana desde fuera
        (por ejemplo con tail -f /tmp/opencode/overlay.log) sin depender de
        herramientas de X11, que en este equipo no están instaladas.
        """
        registro(f"estado: {self.width()}x{self.height()} en {self.x()},{self.y()}")
        self._programar_sonda()

    # --- medida de la muñequita --------------------------------------------
    def _programar_sonda(self) -> None:
        """Mide el tamaño real del avatar dibujado (no el de la ventana).

        La ventana puede ser muy grande y el avatar ocupar solo una parte:
        con escala 1 el modelo se dibuja ajustado al lienzo, asi que su tamaño
        real es min(ancho/natural.ancho, alto/natural.alto) * natural.
        """
        if not hasattr(self, "_temporizador_sonda"):
            self._temporizador_sonda = QTimer(self)
            self._temporizador_sonda.setSingleShot(True)
            self._temporizador_sonda.timeout.connect(self._sonda)
        self._temporizador_sonda.start(300)      # agrupa los cambios seguidos

    def _sonda(self) -> None:
        self.page().runJavaScript(
            "window.__casavita"
            " ? (function () {"
            "     const c = __casavita.canvas(), n = __casavita.natural();"
            "     const e = Math.min(c.ancho / n.ancho, c.alto / n.alto);"
            "     return 'muñequita ' + Math.round(n.ancho * e) + 'x'"
            "          + Math.round(n.alto * e)"
            "          + ' (escala ' + e.toFixed(3) + ')';"
            "   })()"
            " : 'SIN PUENTE'",
            lambda v: (registro(f"sonda: {v}"),
                        self._programar_sonda() if v == "SIN PUENTE" else None),
        )

    # --- ceñir la ventana al tamano del avatar -----------------------------
    def _ceñir(self, alto_deseado: int) -> None:
        """Ajusta el alto de la ventana al avatar.

        La pagina escala el modelo con:
            escala = min(ancho/natural.ancho, alto/natural.alto) * ajuste.escala
        Con escala 1 el modelo llena justo la ventana, sin huecos transparentes.
        """
        if getattr(self, "_ceñir_listo", False):
            return
        self._alto_deseado = alto_deseado
        self.page().runJavaScript(
            "window.__casavita && window.__casavita.natural()", self._con_medidas
        )
        QTimer.singleShot(500, lambda: self._ceñir(alto_deseado))

    def _con_medidas(self, medidas) -> None:
        if not isinstance(medidas, dict):
            # El puente todavia no existe: reintentar.
            if getattr(self, "_intentos", 0) > 40:
                registro("el puente __casavita no apareció")
            self._intentos = getattr(self, "_intentos", 0) + 1
            return
        nat_ancho = float(medidas.get("ancho") or 0)
        nat_alto = float(medidas.get("alto") or 0)
        if nat_ancho <= 0 or nat_alto <= 0:
            registro(f"medidas invalidas: {medidas}")
            return

        alto = self._alto_deseado
        ancho = max(80, int(round(alto * nat_ancho / nat_alto)))
        self.resize(ancho, alto)
        self._ceñir_listo = True
        registro(f"ceñida al avatar: {ancho}x{alto} "
                 f"(modelo natural {nat_ancho:.0f}x{nat_alto:.0f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-barra", action="store_true",
                    help="con barra de titulo (por defecto va sin barra ni bordes)")
    ap.add_argument("--alto-avatar", type=int, default=0,
                    help="alto en pixeles; 0 = usar el ultimo tamaño guardado")
    ap.add_argument("--capa", action="store_true",
                    help="sin barra y a pantalla completa")
    ap.add_argument("--atravesar", action="store_true",
                    help="el raton atraviesa la ventana (solo con --capa)")
    args = ap.parse_args()

    # El tamaño y la posicion elegidos por el usuario se recuerdan: si no se
    # pasa --alto-avatar, se arranca con lo que quedo la ultima vez.
    guardado = leer_posicion() or {}
    if not args.alto_avatar and guardado.get("alto"):
        args.alto_avatar = int(guardado["alto"])
    if not args.alto_avatar:
        args.alto_avatar = 800

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    pantalla = app.primaryScreen().availableGeometry()

    sin_barra = args.capa or not args.con_barra

    if args.capa:
        modo = "capa"
        url = f"{URL_BASE}?escala=0.55&x=0.94&y=0.86"
        ventana = VentanaAvatar(url, not args.atravesar, modo,
                                args.alto_avatar, sin_barra=True)
        ventana.setGeometry(app.primaryScreen().geometry())
    else:
        modo = "ventana"
        # escala 1 = el modelo llena la ventana justa; ancla al centro.
        url = f"{URL_BASE}?escala=1.0&x=0.5&y=0.5"
        ventana = VentanaAvatar(url, True, modo, args.alto_avatar,
                                sin_barra=sin_barra,
                                ancho_inicial=int(args.alto_avatar * 0.8))
        if guardado and guardado.get("modo") == modo:
            # Recupera el tamaño que eligio el usuario la ultima vez.
            if guardado.get("ancho") and guardado.get("alto"):
                ventana.resize(int(guardado["ancho"]), int(guardado["alto"]))
            ventana.move(int(guardado["x"]), int(guardado["y"]))
        else:
            ventana.move(pantalla.right() - ventana.width() - 60,
                         pantalla.bottom() - ventana.height() - 60)

    ventana.show()
    ventana.raise_()
    registro(f"mostrada modo={modo} en {ventana.x()},{ventana.y()} "
             f"({ventana.width()}x{ventana.height()})")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())