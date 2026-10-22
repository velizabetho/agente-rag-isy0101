"""Public SNBP sources used by the project and a downloader for them.

Run: python -m bookclub.data.sources
"""

import urllib.request
from pathlib import Path

from bookclub.config import RAW_DIR

_DRIVE = "https://drive.google.com/uc?export=download&id={}"

GUIDELINES_FILE = "orientaciones-clubes-lectura-2025.pdf"
BPDIGITAL_FILE = "catalogo-bpdigital-clubes.xlsx"

SOURCES: dict[str, str] = {
    GUIDELINES_FILE: "https://www.bibliotecaspublicas.gob.cl/sites/www.bibliotecaspublicas.gob.cl/files/2026-02/orientaciones-clubes-lectura%202025.pdf",
    BPDIGITAL_FILE: _DRIVE.format("1fHFShVyPDAeQhHkKFuifstboCrI1T8Yy"),
    # Regional catalogs for book clubs (bibliotecaspublicas.gob.cl -> Clubes de lectura -> Catálogos)
    "catalogo-arica-y-parinacota.pdf": _DRIVE.format("19dKSNT5GrBfUg3B71zd6MX7ENLAY4cj_"),
    "catalogo-tarapaca.pdf": _DRIVE.format("1lIBw0ssylHI5_FF4io7etYamsyrMnSzm"),
    "catalogo-antofagasta.pdf": _DRIVE.format("1nEeF-18lpUGTUKgfVqhwQXLwURuaPhiW"),
    "catalogo-atacama.pdf": _DRIVE.format("1_XbTOXxJXduovKvA8k2QGa_RxlQZJOAW"),
    "catalogo-coquimbo.pdf": _DRIVE.format("1VhY_iB1cSmyVHwLGq3aBpV8j6aq3W45K"),
    "catalogo-valparaiso.pdf": _DRIVE.format("1jzPVY4h3o5EPZeUw99uyEKoRN554ocCU"),
    "catalogo-metropolitana.pdf": _DRIVE.format("1nFZNWT0Yce3BBRlV4Ihl89vnB4Wnuj6k"),
    "catalogo-ohiggins.pdf": _DRIVE.format("1-fGxG5ZVua_C3NG9kTWliF-BueAxUGoF"),
    "catalogo-maule.pdf": _DRIVE.format("1R2bGq34Xa3U3EbMemQ1RRuK8wFRWCbON"),
    "catalogo-nuble.pdf": _DRIVE.format("1knVDbiKczHtuSLNB2JwwyqQ03wGjk-X_"),
    "catalogo-biobio.pdf": _DRIVE.format("1yC9FHJwWMk_lYBq0GsAmMMLNsgjeySYZ"),
    "catalogo-la-araucania.pdf": _DRIVE.format("1F7UaE41dF6mA5Q44op1JuaCxzCjiPJdY"),
    "catalogo-los-rios.pdf": _DRIVE.format("1uO5dO8aWvucYZAJ3x6b61uxlW8V4EScF"),
    "catalogo-los-lagos.pdf": _DRIVE.format("1KVp6LpMhl6pCSjN95R7BKqaB1LK4lp2n"),
    "catalogo-aysen.pdf": _DRIVE.format("1fM3Bk3p1-u0n-BvI27Jy2Tkf1Z_R0oNG"),
    "catalogo-magallanes.pdf": _DRIVE.format("1Qr66NEupqbmvHVrquxQ4IwEyqLmrFHV6"),
}


def download_all(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = raw_dir / name
        if target.exists() and not force:
            print(f"exists      {name}")
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
        print(f"downloaded  {name} ({target.stat().st_size // 1024} KB)")
