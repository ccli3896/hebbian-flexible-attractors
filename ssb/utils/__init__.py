"""Utilities: simulator, config, batch initialization, stimulus broadcasting.

Submodules are imported lazily so individual files can be used
without requiring every other module to be complete (e.g., the
``stimulus`` module is testable without ``batch_inits`` having been
filled in yet).
"""
