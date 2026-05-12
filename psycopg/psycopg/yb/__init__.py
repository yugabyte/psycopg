"""
YugabyteDB smart-driver subpackage.

All smart-driver behaviour (cluster discovery, connection-count tracking,
least-loaded picking, topology filtering) lives under this subpackage. Edits
to upstream files (connection_async.py, _conninfo_attempts_async.py) only
call into here — they don't carry smart-driver logic themselves.

This keeps the merge-conflict surface small on each upstream rebase.
"""

# Copyright (C) 2026 Yugabyte
