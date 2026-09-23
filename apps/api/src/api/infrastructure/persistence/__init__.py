"""Persistence adapters.

Two implementations of the same domain ports:

* `memory` -- in-process, used by tests and the contract suite.
* `sql` -- SQLAlchemy over PostgreSQL.

Both are exercised by one shared contract suite, so the in-memory fake cannot
drift from the real store.
"""
