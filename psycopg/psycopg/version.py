"""
psycopg-yugabytedb distribution version file.
"""

# Copyright (C) 2020 The Psycopg Team
# Modified for psycopg-yugabytedb fork

from importlib import metadata

try:
    __version__ = metadata.version("psycopg-yugabytedb")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0.0"
