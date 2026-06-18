#!/usr/bin/env python3

# Create the psycopg-yugabytedb-binary package by renaming and patching psycopg-yugabytedb-c.
#
# Note for fork maintainers: only the *distribution* name changes (psycopg-yugabytedb-c
# -> psycopg-yugabytedb-binary). The on-disk Python package name still flips from
# psycopg_c to psycopg_binary so the import surface matches upstream's pattern;
# our driver imports `from psycopg_c import pq` and `from psycopg_binary import pq`
# unchanged, because the C extension itself isn't modified.

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

curdir = Path(__file__).parent
pdir = curdir / "../.."

if (target := (pdir / "psycopg_binary")).exists():
    raise Exception(f"path {target} already exists")


def sed_i(pattern: str, repl: str, filename: str | Path) -> None:
    with open(filename, "rb") as f:
        data = f.read()

    if (newdata := re.sub(pattern.encode("utf8"), repl.encode("utf8"), data)) != data:
        with open(filename, "wb") as f:
            f.write(newdata)


shutil.copytree(pdir / "psycopg_c", target)
shutil.move(str(target / "psycopg_c"), str(target / "psycopg_binary"))
shutil.move(str(target / "README-binary.rst"), str(target / "README.rst"))
# Distribution name: psycopg-yugabytedb-c -> psycopg-yugabytedb-binary.
sed_i("psycopg-yugabytedb-c", "psycopg-yugabytedb-binary", target / "pyproject.toml")
sed_i("psycopg-yugabytedb-c", "psycopg-yugabytedb-binary", target / "psycopg_binary/version.py")
# Python package name: psycopg_c -> psycopg_binary (unchanged from upstream).
sed_i(r'"psycopg_c([\./][^"]+)?"', r'"psycopg_binary\1"', target / "pyproject.toml")
sed_i(r"__impl__\s*=.*", '__impl__ = "binary"', target / "psycopg_binary/pq.pyx")
for dirpath, dirnames, filenames in os.walk(target):
    for filename in filenames:
        if os.path.splitext(filename)[1] not in (".pyx", ".pxd", ".py"):
            continue
        sed_i(r"\bpsycopg_c\b", "psycopg_binary", Path(dirpath) / filename)
