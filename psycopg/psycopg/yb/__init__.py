"""
YugabyteDB smart-driver subpackage.

All smart-driver behaviour (cluster discovery, connection-count tracking,
least-loaded picking, topology filtering) lives under this subpackage. Edits
to upstream files (connection_async.py, _conninfo_attempts_async.py) only
call into here — they don't carry smart-driver logic themselves.

This keeps the merge-conflict surface small on each upstream rebase.

Logging
-------

Every smart-driver module uses ``logging.getLogger(__name__)``, so the
loggers form a tree rooted at ``psycopg.yb``:

  * ``psycopg.yb.registry``           — bootstrap, refresh, counters, control conn,
                                         FailoverGroup pairing + operator API
  * ``psycopg.yb.discovery``          — yb_servers() query
  * ``psycopg.yb.policy.base``        — eligibility filter
  * ``psycopg.yb.policy.cluster_aware``  — least-loaded pick, tie-break
  * ``psycopg.yb.policy.topology_aware`` — placement filter
  * ``psycopg.yb.health``             — cluster_status_check stub + transitions
                                         (will host the tracker-table check in a
                                         follow-on patch — currently a no-op stub)
  * ``psycopg.yb.health_probe``       — xCluster background probe thread
                                         (INFO on transitions, DEBUG per tick,
                                         WARNING on probe exceptions)
  * ``psycopg.yb.pool``               — pool ``check=`` callback that evicts
                                         conns belonging to the inactive cluster
  * ``psycopg.yb.dispatcher``         — connect-time branch on
                                         ``FailoverGroup.status`` to route to
                                         primary or secondary cluster

Tune verbosity per subsystem, or set the parent ``psycopg.yb`` once:

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("psycopg.yb").setLevel(logging.DEBUG)

The subpackage also registers a custom ``TRACE`` level (numeric 5, below
``DEBUG``). Use it via the standard ``logger.log(TRACE, …)`` API, or via
the ``logger.trace(…)`` helper we install on ``logging.Logger``:

    from psycopg.yb import TRACE
    logging.getLogger("psycopg.yb").setLevel(TRACE)

Level usage convention inside the subpackage:

  * WARNING — driver had to give up (all candidates refused, …)
  * INFO    — lifecycle events worth seeing by default (cluster bootstrap,
              host quarantined, control conn moved, topology change observed)
  * DEBUG   — per-operation breadcrumbs (each pick, each refresh result,
              each control-conn open/close)
  * TRACE   — very verbose (every counter mutation, every filtered candidate)
"""

# Copyright (C) 2026 Yugabyte

import logging as _logging


# Custom log level finer-grained than DEBUG. Standard library levels are
# CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, NOTSET=0; we slot
# in at 5. Re-registering the name is a no-op if it's already present.
TRACE = 5
_logging.addLevelName(TRACE, "TRACE")


# Install `logger.trace(...)` as a convenience so call sites read like the
# other levels. We only install if no other library beat us to it, so we
# don't clobber a third-party `trace` method that might use it differently.
if not hasattr(_logging.Logger, "trace"):
    def _trace(self, message, *args, **kwargs):  # type: ignore[no-redef]
        if self.isEnabledFor(TRACE):
            self._log(TRACE, message, args, **kwargs)
    _logging.Logger.trace = _trace  # type: ignore[attr-defined]

del _logging
