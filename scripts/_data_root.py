"""
_data_root.py
=============
Localisation du dossier de données expérimentales, sans chemin absolu.

Les trajectoires réelles ne sont PAS versionnées : elles appartiennent à
l'équipe de biologie et sortent du périmètre de ce dépôt. Les scripts qui les
lisent doivent donc pouvoir les trouver sans supposer l'arborescence d'une
machine particulière. Par ordre de priorité :

  1. l'option `--data-root` du script ;
  2. la variable d'environnement `ANTSIM_DATA` ;
  3. `../Data` relativement à la racine du dépôt — l'emplacement du dossier
     dans l'organisation d'origine du projet.
"""

import os
from pathlib import Path

#: Racine du dépôt (le dossier qui contient `antsim/`, `scripts/`, ...).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Emplacement supposé des données dans l'organisation d'origine du projet.
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "Data"


def data_root(override=None):
    """Renvoie le dossier de données, ou lève une erreur explicite."""
    for candidate in (override, os.environ.get("ANTSIM_DATA"), DEFAULT_DATA_ROOT):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise SystemExit(
        "Dossier de données introuvable.\n"
        "Les trajectoires expérimentales ne sont pas versionnées avec le code.\n"
        "Indiquer leur emplacement par --data-root, par la variable "
        "d'environnement ANTSIM_DATA,\n"
        f"ou les placer dans {DEFAULT_DATA_ROOT}.")


def add_argument(parser):
    """Ajoute l'option `--data-root` à un parseur d'arguments."""
    parser.add_argument("--data-root", default=None,
                        help="dossier des trajectoires expérimentales "
                             "(défaut : $ANTSIM_DATA, sinon ../Data)")
