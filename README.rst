YugabyteDB Psycopg 3 -- Database adapter for Python
===================================================

Psycopg 3 is a modern implementation of a PostgreSQL adapter for Python.
This is the YugabyteDB smart-driver fork, distributed as ``psycopg-yugabytedb``.


YugabyteDB smart driver
-----------------------

This fork (distribution name ``psycopg-yugabytedb``) extends psycopg 3 with
cluster-aware and topology-aware connection load balancing for `YugabyteDB`_.
The import name stays ``psycopg`` — existing code is unchanged. Adding
``load_balance_hosts=true`` to the conninfo string opts into the smart driver.

.. _YugabyteDB: https://www.yugabyte.com/

Opt in with one parameter:

.. code-block:: python

    import psycopg

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true"
    )

The driver discovers every live tserver via ``yb_servers()`` on the first
connect, then distributes subsequent connects across them with a least-loaded
picker (random tie-break, atomic under a per-cluster lock). One contact point
in ``host=…`` is enough to bootstrap; the rest of the cluster is discovered.

Smart-driver conninfo parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 14 12 44

   * - Parameter
     - Values
     - Default
     - Description
   * - ``load_balance_hosts``
     - ``true`` / ``false`` /
       ``disable`` / ``random``
     - absent
     - ``true`` enables the smart driver. ``false`` is equivalent to libpq's
       ``disable``. ``disable`` / ``random`` pass through to libpq unchanged.
   * - ``topology_keys``
     - ``cloud.region.zone``
       (comma-separated for
       multiple keys)
     - none
     - Restrict picks to tservers matching at least one placement. Zone may
       be ``*`` for any zone in that cloud/region. Cloud / region wildcards
       are rejected. Strict in v1 — no cluster-wide fallback if all
       matching nodes are down.
   * - ``yb_servers_refresh_interval``
     - seconds (integer)
     - ``300``
     - How often to re-query ``yb_servers()``. Clamped to ``[0, 600]``.
   * - ``failed_host_reconnect_delay_secs``
     - seconds (integer)
     - ``5``
     - How long to quarantine a node after a failed connect, before
       reconsidering it. Clamped to ``[0, 60]``.

Topology-aware example — bind traffic to one zone:

.. code-block:: python

    conn = psycopg.connect(
        "host=h1,h2,h3 port=5433 user=yugabyte dbname=yugabyte "
        "load_balance_hosts=true "
        "topology_keys=cloud1.datacenter1.zoneA"
    )

The upstream ``psycopg-pool`` connection pool works unchanged with the smart
driver — the dispatcher sits underneath the pool, so pool-managed connections
honour the configured policy too.

The fork is `pre-release`_. Install with ``pip install --pre psycopg-yugabytedb``
or pin to ``3.3.4.1rc1``. It cannot coexist with upstream ``psycopg`` in the
same environment (both install into ``site-packages/psycopg/``).

.. _pre-release: https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases


Installation
------------

Quick version::

    pip install "psycopg[binary,pool]"

For further information about installation please check `the documentation`__.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html


.. _Hacking:

Hacking
-------

In order to work on the Psycopg source code, you must have the
``libpq`` PostgreSQL client library installed on the system. For instance, on
Debian systems, you can obtain it by running::

    sudo apt install libpq5

On macOS, run::

    brew install libpq

On Windows you can use EnterpriseDB's `installers`__ to obtain ``libpq``
which is included in the Command Line Tools.

.. __: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

You can then clone this repository to develop Psycopg::

    git clone https://github.com/psycopg/psycopg.git
    cd psycopg

Please note that the repository contains the source code of several Python
packages, which may have different requirements:

- The ``psycopg`` directory contains the pure python implementation of
  ``psycopg``. The package has only a runtime dependency on the ``libpq``, the
  PostgreSQL client library, which should be installed in your system.

- The ``psycopg_c`` directory contains an optimization module written in
  C/Cython. In order to build it you will need a few development tools: please
  look at `Local installation`__ in the docs for the details.

- The ``psycopg_pool`` directory contains the `connection pools`__
  implementations. This is kept as a separate package to allow a different
  release cycle.

.. __: https://www.psycopg.org/psycopg3/docs/basic/install.html#local-installation
.. __: https://www.psycopg.org/psycopg3/docs/advanced/pool.html

You can create a local virtualenv and install the packages `in
development mode`__, together with their development and testing
requirements::

    python -m venv .venv
    source .venv/bin/activate

    # Install the base Psycopg package in editable mode
    pip install --config-settings editable_mode=strict -e "./psycopg[dev,test]"

    # Install the connection pool package in editable mode
    pip install --config-settings editable_mode=strict -e ./psycopg_pool

    # Install the C speedup extension
    pip install ./psycopg_c

.. __: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs

The ``--config-settings editable_mode=strict`` will be probably required
to work around the problem of the `editable mode broken`__.

.. __: https://github.com/pypa/setuptools/issues/3557

Now hack away! You can run the tests using::

    psql -c 'create database psycopg_test'
    export PSYCOPG_TEST_DSN="dbname=psycopg_test"
    pytest

The project includes some `pre-commit`__ hooks to check that the code is valid
according to the project coding convention. Please make sure to install them
by running::

    pre-commit install

This will allow to check lint errors before submitting merge requests, which
will save you time and frustrations.

.. __: https://pre-commit.com/


Cross-compiling
---------------

To use cross-platform zipapps created with `shiv`__ that include Psycopg
as a dependency you must also have ``libpq`` installed. See
`the section above <Hacking_>`_ for install instructions.

.. __: https://github.com/linkedin/shiv
