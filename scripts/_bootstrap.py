"""Ajoute la racine du projet au chemin d'import.

Permet de lancer les scripts sans installer le paquet (`python scripts/x.py`).
Devient sans effet si le paquet a été installé (`pip install -e .`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
