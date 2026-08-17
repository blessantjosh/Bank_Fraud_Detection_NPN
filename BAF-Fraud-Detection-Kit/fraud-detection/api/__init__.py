"""
fraud-detection/api -- security-hardened FastAPI service in front of the ML
fraud-detection pipeline in src/.

This package is deliberately isolated from src/ (imported from, never
modified) so it can be developed in parallel with the ML pipeline work
happening there. See api/SECURITY.md for the full security posture and
api/README.md for how to run it.
"""
