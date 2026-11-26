"""Input and output guardrails of the agent.

Input (before any LLM call):
- Length limit: very long messages are rejected (prompt stuffing, unbounded consumption).
- Prompt injection: Llama Prompt Guard 2 (served by Groq) scores the message; above the threshold the turn is
  blocked. If the classifier is unavailable a keyword heuristic is used and the result is marked as degraded.
- Personal data: RUT, e-mail and phone numbers are masked, so they never reach the LLM provider, the
  long-term memory, the session notes or the logs.

Output (before the answer reaches the user): personal data is masked again and answers that leak the system
prompt are replaced.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from bookclub.config import get_settings
from bookclub.observability.recorder import span

MAX_INPUT_CHARS = 2000
INJECTION_THRESHOLD = 0.8
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"

BLOCKED_REPLY = {
    "too_long": f"Tu mensaje es muy largo (máximo {MAX_INPUT_CHARS} caracteres). ¿Puedes resumirlo?",
    "prompt_injection": "No puedo seguir instrucciones que intenten cambiar mis reglas o revelar mi configuración. "
                        "Puedo ayudarte con el club: lecturas, reservas, calendario, sesiones o el manual del SNBP.",
}
LEAK_REPLY = "No puedo compartir mi configuración interna. ¿En qué te ayudo con el club?"

_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "rut": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?56[\s-]?)?9[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
}
_PII_LABELS = {"email": "[CORREO]", "rut": "[RUT]", "phone": "[TELÉFONO]"}

# Heuristic used only when the classifier cannot be reached (normalized: lowercase, no accents).
_INJECTION_PATTERNS = re.compile(
    r"ignora\w* (todas? )?(las |tus )?(instrucciones|reglas)|olvida\w* (todas? )?(las |tus )?(instrucciones|reglas)"
    r"|ignore (all |previous |the )*(instructions|rules)|system prompt|prompt (de|del) sistema"
    r"|(muestra|revela|repite)\w* (tu|el) (prompt|configuracion|instrucciones)|modo desarrollador|jailbreak|\bdan\b"
)
# Fragments of the system prompt that must never appear in an answer.
_LEAK_MARKERS = ("## Cuándo usar cada herramienta", "## Reglas", "## Plan de tareas para esta solicitud")


@dataclass
class InputCheck:
    allowed: bool
    text: str  # message to pass on (PII masked)
    reason: str | None = None
    injection_score: float | None = None
    classifier: str = "prompt_guard"  # prompt_guard | heuristic (degraded)
    pii: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("text")
        return data


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace RUT, e-mail and phone numbers with placeholders; return the masked text and the types found."""
    found = []
    for kind, pattern in _PII_PATTERNS.items():
        text, n = pattern.subn(_PII_LABELS[kind], text)
        if n:
            found.append(kind)
    return text, found


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def heuristic_injection_score(text: str) -> float:
    return 1.0 if _INJECTION_PATTERNS.search(_normalize(text)) else 0.0


@lru_cache
def _groq_client():
    from groq import Groq

    return Groq(api_key=get_settings().groq_api_key, max_retries=2, timeout=10)


def prompt_guard_score(text: str) -> float:
    """Probability (0-1) that the text is a prompt injection / jailbreak, according to Llama Prompt Guard 2."""
    response = _groq_client().chat.completions.create(
        model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text[-MAX_INPUT_CHARS:]}]
    )
    return float(response.choices[0].message.content)


def check_input(text: str, classifier: Callable[[str], float] = prompt_guard_score) -> InputCheck:
    if len(text) > MAX_INPUT_CHARS:
        return InputCheck(False, text, reason="too_long")
    with span("prompt_guard", kind="guardrail") as s:
        try:
            score, source = classifier(text), "prompt_guard"
        except Exception as exc:  # classifier down or rate limited: fall back, do not let the turn through unchecked
            s["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            score, source = heuristic_injection_score(text), "heuristic"
        s["score"] = round(score, 4)
    masked, pii = mask_pii(text)
    if score >= INJECTION_THRESHOLD:
        return InputCheck(False, masked, "prompt_injection", score, source, pii)
    return InputCheck(True, masked, None, score, source, pii)


def check_output(text: str) -> tuple[str, list[str]]:
    """Masked answer and the issues found ("pii", "prompt_leak")."""
    if any(marker in text for marker in _LEAK_MARKERS):
        return LEAK_REPLY, ["prompt_leak"]
    masked, pii = mask_pii(text)
    return masked, ["pii"] if pii else []
