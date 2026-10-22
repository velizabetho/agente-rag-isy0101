"""Parse the SNBP regional book-club catalogs (PDF) into structured records.

Every catalog shares the same template: a bold title, one or more author lines
and six labelled fields ("Editorial:", "ISBN:", ...). Records are laid out in
columns, so text is grouped into left-aligned line segments and each segment
is attached to the closest title above it that starts at the same x.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

REGIONS: dict[str, str] = {
    "arica-y-parinacota": "Arica y Parinacota",
    "tarapaca": "Tarapacá",
    "antofagasta": "Antofagasta",
    "atacama": "Atacama",
    "coquimbo": "Coquimbo",
    "valparaiso": "Valparaíso",
    "metropolitana": "Metropolitana",
    "ohiggins": "O'Higgins",
    "maule": "Maule",
    "nuble": "Ñuble",
    "biobio": "Biobío",
    "la-araucania": "La Araucanía",
    "los-rios": "Los Ríos",
    "los-lagos": "Los Lagos",
    "aysen": "Aysén",
    "magallanes": "Magallanes",
}

# Label prefix -> CatalogEntry field name.
_LABELS = {
    "editorial:": "publisher",
    "isbn:": "isbn",
    "publicación:": "year",
    "páginas:": "pages",
    "copias disponibles:": "copies",
    "género:": "genre",
}
_TITLE_SIZE = (13.5, 14.5)  # Gotham-Bold 14 pt
_FOOTER_MARGIN = 45.0  # section footer and page number live in the bottom band
_X_TOLERANCE = 3.0  # max x offset between a title and its record lines
_GAP = 12.0  # max horizontal gap between spans of the same segment
_LABEL_RE = re.compile("(" + "|".join(map(re.escape, _LABELS)) + ")", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogEntry:
    region: str
    title: str
    author: str | None
    publisher: str | None
    isbn: str | None
    year: int | None
    pages: int | None
    copies: int | None
    genre: str | None
    source_page: int


@dataclass
class _Segment:
    x: float
    y: float
    x1: float
    text: str
    is_title: bool


def _clean(text: str) -> str:
    return " ".join(text.replace("\xad", "").split())


def _segments(page: pymupdf.Page) -> list[_Segment]:
    """Return left-aligned text segments (contiguous spans on one baseline)."""
    footer_y = page.rect.height - _FOOTER_MARGIN
    spans = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip() and span["origin"][1] < footer_y:
                    spans.append(span)
    spans.sort(key=lambda s: s["origin"][0])

    segments: list[_Segment] = []
    for span in spans:
        x, y = span["origin"]
        is_title = "Bold" in span["font"] and _TITLE_SIZE[0] <= span["size"] <= _TITLE_SIZE[1]
        prev = next(
            (
                s
                for s in segments
                if abs(s.y - y) < 1.0 and -_GAP / 2 <= x - s.x1 <= _GAP and s.is_title == is_title
            ),
            None,
        )
        if prev:
            prev.text += span["text"]
            prev.x1 = span["bbox"][2]
        else:
            segments.append(_Segment(x, y, span["bbox"][2], span["text"], is_title))
    return segments


def _group_records(segments: list[_Segment]) -> list[list[_Segment]]:
    """Attach each segment to the nearest title above it in the same column."""
    titles = [s for s in segments if s.is_title]
    # A title wrapped over several lines starts one record only.
    heads = [
        t
        for t in titles
        if not any(
            o is not t and abs(o.x - t.x) < _X_TOLERANCE and 0 < t.y - o.y < 25 for o in titles
        )
    ]
    groups: dict[int, list[_Segment]] = {id(h): [] for h in heads}
    for seg in segments:
        owners = [h for h in heads if abs(h.x - seg.x) < _X_TOLERANCE and h.y <= seg.y + 0.5]
        if owners:
            groups[id(max(owners, key=lambda h: h.y))].append(seg)
    return [groups[id(h)] for h in heads]


def _int(text: str | None) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None


def _year(text: str | None) -> int | None:
    """Return a standalone 4-digit year; malformed values such as '20162' give None."""
    match = re.search(r"\b(?:1[5-9]|20)\d{2}\b", text or "")
    return int(match.group()) if match else None


def _isbn(text: str | None) -> str | None:
    """Keep digits and X; placeholders such as 'No posee' or '--' give None."""
    value = re.sub(r"[^0-9X]", "", (text or "").upper())
    return value if any(c.isdigit() for c in value) else None


def _join_authors(lines: list[str]) -> str | None:
    """Join author lines; role-tagged lines like 'X (texto)' are separated by '; '."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif line.startswith("("):
            out += " " + line
        else:
            out += ("; " if out.endswith(")") else " ") + line
    return out or None


def _parse_record(segments: list[_Segment], region: str, page_no: int) -> CatalogEntry | None:
    title_lines: list[str] = []
    author_lines: list[str] = []
    fields: dict[str, str] = {}
    current: str | None = None
    for seg in sorted(segments, key=lambda s: s.y):
        text = _clean(seg.text)
        if _LABEL_RE.match(text):
            # A line may hold several "Label: value" pairs (typo in some PDFs).
            parts = _LABEL_RE.split(text)[1:]
            for label, value in zip(parts[::2], parts[1::2]):
                current = _LABELS[label.lower()]
                fields[current] = value.strip()
        elif seg.is_title and not author_lines and current is None:
            title_lines.append(text)
        elif current is None:
            author_lines.append(text)
        else:  # wrapped field value
            fields[current] = f"{fields[current]} {text}".strip()
    if not fields or not title_lines:
        return None
    return CatalogEntry(
        region=region,
        title=" ".join(title_lines),
        author=_join_authors(author_lines),
        publisher=fields.get("publisher") or None,
        isbn=_isbn(fields.get("isbn")),
        year=_year(fields.get("year")),
        pages=_int(fields.get("pages")),
        copies=_int(fields.get("copies")),
        genre=fields.get("genre") or None,
        source_page=page_no,
    )


def parse_regional_catalog(path: Path, region: str) -> list[CatalogEntry]:
    """Parse one regional catalog PDF; `region` is stored as given (display name)."""
    entries: list[CatalogEntry] = []
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for group in _group_records(_segments(page)):
                entry = _parse_record(group, region, page_no)
                if entry:
                    entries.append(entry)
    return entries


def parse_all(raw_dir: Path) -> list[CatalogEntry]:
    """Parse every `catalogo-<slug>.pdf` found in `raw_dir`, in REGIONS order."""
    entries: list[CatalogEntry] = []
    for slug, name in REGIONS.items():
        path = raw_dir / f"catalogo-{slug}.pdf"
        if path.exists():
            entries.extend(parse_regional_catalog(path, name))
    return entries


if __name__ == "__main__":
    from collections import defaultdict

    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    by_region: dict[str, list[CatalogEntry]] = defaultdict(list)
    for e in parse_all(raw):
        by_region[e.region].append(e)

    print(f"{'region':<20}{'records':>8}{'no title':>9}{'no isbn':>8}{'no copies':>10}")
    for name, rows in by_region.items():
        print(
            f"{name:<20}{len(rows):>8}"
            f"{sum(not r.title for r in rows):>9}"
            f"{sum(r.isbn is None for r in rows):>8}"
            f"{sum(r.copies is None for r in rows):>10}"
        )
    print(f"{'TOTAL':<20}{sum(map(len, by_region.values())):>8}")
    for name, rows in by_region.items():
        print(f"\n== {name}")
        step = max(len(rows) // 3, 1)
        for r in rows[::step][:3]:
            print(f"  {r}")
