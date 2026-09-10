psycopg-yugabytedb: YugabyteDB smart-driver — binary package
============================================================

This distribution package is an optional component of
`psycopg-yugabytedb`__: it contains the C-accelerated
``psycopg_binary`` package, pre-compiled with a bundled ``libpq`` so
no build tools or PostgreSQL client library are required on the target
machine.

.. __: https://pypi.org/project/psycopg-yugabytedb/

You shouldn't install this package directly. Use instead ::

    pip install "psycopg-yugabytedb[binary]"

which pulls in a binary matching the pure-Python ``psycopg-yugabytedb``
version installed. Installing this package requires pip >= 20.3.

Wheels are published for the following platforms:

* macOS 11+ on Apple Silicon (arm64)
* Linux x86_64 (manylinux_2_17 / manylinux2014)
* CPython 3.10, 3.11, 3.12, 3.13, 3.14

For other platforms fall back to installing ``psycopg-yugabytedb`` and
letting it compile against the system ``libpq`` (requires
``libpq-dev`` / ``libpq-devel`` and a C toolchain).

About the fork
--------------

``psycopg-yugabytedb`` is a fork of `psycopg 3`__ that adds
cluster-aware and topology-aware connection load balancing for
`YugabyteDB`__. Application code doesn't change — the import name
stays ``psycopg``; adding ``load_balance_hosts=true`` to the conninfo
string opts into the smart-driver behaviour.

The C extension in this binary distribution is unmodified from
upstream; only the packaging metadata and bundled dependencies change.

.. __: https://pypi.org/project/psycopg/
.. __: https://www.yugabyte.com/

Documentation
-------------

* Fork readme and usage examples:
  https://github.com/yugabyte/psycopg#readme
* Upstream psycopg documentation (API reference):
  https://www.psycopg.org/psycopg3/docs/


Copyright (C) 2020 The Psycopg Team
Copyright (C) 2026 Yugabyte
