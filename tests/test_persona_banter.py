"""Regresion de los dos arreglos del prompt de Casavita.

1) `ChatConfig.banter: false` tiene que quitar las reglas de guasa del prompt
   (antes estaban fijas en chat_turn y el interruptor no hacia nada).
2) El check-in usa las notas del dueño, y esas notas deben GUARDAR lo que dijo
   casave (intent.chat.text), no lo que dice Casavita (`full`).

Run with:  python -m pytest tests/ -v
"""
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from config import ChatConfig, PersonaConfig
from core.persona import Persona


@pytest.fixture()
def persona() -> Persona:
    return Persona(PersonaConfig(name="Casavita"))


# ─────────────────────────────────────────────────────────────────────
# interruptor chat.banter
# ─────────────────────────────────────────────────────────────────────

def test_chat_config_banter_por_defecto_sigue_activado():
    assert ChatConfig().banter is True


def test_banter_true_conserva_las_reglas_de_guasa(persona: Persona):
    turno = persona.chat_turn(
        username="fulano", platform="twitch", text="hola", is_highlight=False,
        chat_banter=True,
    )
    assert "BANTER RULES" in turno
    assert "NO BANTER" not in turno
    # las reglas siguen siendo las de guasar, no la prohibicion
    assert "roast" in turno


def test_banter_false_sustituye_las_reglas_por_prohibicion_de_guasa(persona: Persona):
    turno = persona.chat_turn(
        username="fulano", platform="twitch", text="hola", is_highlight=False,
        chat_banter=False,
    )
    assert "NO BANTER" in turno
    assert "BANTER RULES" not in turno
    # ni "roast" como permiso, ni "teasing" permitido
    assert "you are allowed to roast" not in turno
    assert "You may tease him" not in turno


def test_banter_false_no_altera_el_resto_del_turno(persona: Persona):
    kwargs = dict(
        username="fulano", platform="twitch", text="hola", is_highlight=False,
    )
    con = persona.chat_turn(**kwargs, chat_banter=True)
    sin = persona.chat_turn(**kwargs, chat_banter=False)
    # el mensaje del chat entra igual en ambos
    assert con.split('"hola"')[0] == sin.split('"hola"')[0]
    for trozo in (
        "PLAIN TEXT ONLY",
        "HOW TO ANSWER",
        "Do not break the flow of the stream.",
    ):
        assert trozo in con and trozo in sin


def test_check_in_sin_banter_no_invita_a_tocar_las_narices_de_casave(persona: Persona):
    comun = dict(owner_recent=["casave esta haciendo pipas"])
    con = persona.monologue_turn(check_in=True, chat_banter=True, **comun)
    sin = persona.monologue_turn(check_in=True, chat_banter=False, **comun)
    assert "You may tease him a little" in con
    assert "You may tease him a little" not in sin
    assert "no teasing" in sin
    # el resto del check-in sigue igual
    assert "Lo último que dijo casave fue:" in con
    assert "Lo último que dijo casave fue:" in sin


# ─────────────────────────────────────────────────────────────────────
# notas del dueño (check-in)
# ─────────────────────────────────────────────────────────────────────

def test_check_in_cita_las_notas_del_dueno():
    p = Persona(PersonaConfig(name="Casavita"))
    turno = p.monologue_turn(check_in=True, owner_recent=["estoy con el microphone roto"])
    assert "Lo último que dijo casave fue:" in turno
    assert "estoy con el microphone roto" in turno
    # solo las 3 ultimas
    turno = p.monologue_turn(
        check_in=True, owner_recent=["a", "b", "c", "d", "e"],
    )
    assert "d / e" in turno
    assert " a / " not in turno


def test_check_in_sin_notas_no_inventa_nada():
    p = Persona(PersonaConfig(name="Casavita"))
    turno = p.monologue_turn(check_in=True, owner_recent=None)
    assert "Lo último que dijo casave fue:" not in turno
