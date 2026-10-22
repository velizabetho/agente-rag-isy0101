"""Enrich the SNBP book-club list with data scraped from each club's ficha page.

Privacy: mediator names and emails present on the fichas are never extracted or stored.
"""

from __future__ import annotations

import html
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

USER_AGENT = "bookclub-student-project/0.1 (educational use)"
TIMEOUT_S = 20
MAX_WORKERS = 4
RETRIES = 1

# Normalized ficha label -> output field. Only whitelisted fields are kept (no mediator/email).
FICHA_FIELDS = {
    "cantidad de participantes": "participantes_texto",
    "frecuencia de sesiones": "frecuencia_texto",
    "n de titulos leidos el 2024": "titulos_texto",
}

NUMBER_WORDS = {"una": 1, "un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "1": 1, "2": 2, "3": 3, "4": 4}
WEEKDAYS = "lunes|martes|miercoles|jueves|viernes|sabados?|domingos?"

_ROW_RE = re.compile(r"<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation noise, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", cell))).strip()


def club_id_from_url(url: str) -> str:
    """Stable slug: last path segment of the ficha URL."""
    return urlparse(url).path.strip("/").split("/")[-1]


def parse_ficha(page: str) -> dict[str, str]:
    """Extract whitelisted fields from a ficha's HTML table."""
    out = {}
    for label, value in _ROW_RE.findall(page):
        field = FICHA_FIELDS.get(_normalize(_clean_cell(label)))
        if field:
            out[field] = _clean_cell(value)
    return out


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


def sessions_per_month(text: str | None) -> float | None:
    """Normalize free-text session frequency to sessions per month (None if unknown)."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = _normalize(text)
    times = r"(una|un|uno|dos|tres|cuatro|[1-4]) (?:vez|veces)"
    if m := re.search(times + r" (?:al|por el|cada|por) mes", t):
        return float(NUMBER_WORDS[m.group(1)])
    if m := re.search(times + r" (?:a la|por|cada) semana", t):
        return float(NUMBER_WORDS[m.group(1)] * 4)
    if re.search(r"quincenal|cada (?:15|quince) dias|cada (?:dos|2) semanas", t):
        return 2.0
    if re.search(rf"semanal|todas las semanas|cada semana|todos los (?:{WEEKDAYS})\b", t):
        return 4.0
    if re.search(r"mensual|cada mes|una vez al mes", t):
        return 1.0
    return None


def _first_int(text: str | None) -> int | None:
    m = re.search(r"\d+", text) if isinstance(text, str) else None
    return int(m.group()) if m else None


def enrich_clubs(csv_in: Path, csv_out: Path) -> pd.DataFrame:
    """Fetch every club ficha and write an enriched CSV; returns the enriched frame.

    Fetch failures are recorded in ``df.attrs["failures"]`` as ``(url, error)`` pairs.
    """
    df = pd.read_csv(csv_in)
    urls = df["url_ficha"].tolist()

    def task(url: str) -> dict[str, str] | Exception:
        try:
            return parse_ficha(fetch(url))
        except Exception as exc:  # noqa: BLE001 - reported to caller
            return exc

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(task, urls))

    failures = [(u, repr(r)) for u, r in zip(urls, results) if isinstance(r, Exception)]
    fichas = [r if isinstance(r, dict) else {} for r in results]

    df.insert(0, "club_id", [club_id_from_url(u) for u in urls])
    df["participantes_texto"] = [f.get("participantes_texto") for f in fichas]
    df["frecuencia_texto"] = [f.get("frecuencia_texto") for f in fichas]
    df["titulos_leidos_2024"] = pd.array([_first_int(f.get("titulos_texto")) for f in fichas], dtype="Int64")
    freq_source = df["frecuencia_texto"].where(df["frecuencia_texto"].notna(), df["frecuencia"])
    df["sesiones_por_mes"] = [sessions_per_month(t) for t in freq_source]
    from_ficha = pd.Series([_first_int(t) for t in df["participantes_texto"]], index=df.index, dtype="Float64")
    df["participantes"] = df["participantes"].astype("Float64").fillna(from_ficha).astype("Int64")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    df.attrs["failures"] = failures
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    csv_in = root / "data" / "clubes_lectura_snbp_2026.csv"
    csv_out = root / "data" / "clubs_enriched.csv"

    before = pd.read_csv(csv_in)
    after = enrich_clubs(csv_in, csv_out)

    print(f"rows: {len(after)} -> {csv_out}")
    print(f"fetch failures: {len(after.attrs['failures'])}")
    for url, err in after.attrs["failures"]:
        print(f"  {url}: {err}")
    before_freq = sum(sessions_per_month(t) is None for t in before["frecuencia"])
    print("missing values (before -> after):")
    print(f"  participantes:       {before['participantes'].isna().sum()} -> {after['participantes'].isna().sum()}")
    print(f"  sesiones_por_mes:    {before_freq} -> {after['sesiones_por_mes'].isna().sum()}")
    print(f"  titulos_leidos_2024: {len(before)} -> {after['titulos_leidos_2024'].isna().sum()}")
