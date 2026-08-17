"""
api/rate_limit.py -- brute-force / abuse protection via slowapi.

Keyed by client IP (slowapi's default get_remote_address). This is an
application-level control; a real deployment should ALSO rate-limit at the
edge (reverse proxy / WAF) -- see api/SECURITY.md "Rate limiting".
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
