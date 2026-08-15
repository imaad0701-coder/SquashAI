"""Validation-only harness for testing the EXISTING, unmodified generic
pose/biomechanics engine against real video (originally built to answer:
"can the current engine reliably measure a backhand movement?").

Nothing in this package is part of the shipped engine. It composes
already-implemented, untouched engine components (engine.pipelines,
engine.biomechanics, engine.tracking, ...) and reports what they produce.
It performs NO shot classification and adds NO new biomechanics algorithms,
landmarks, benchmarks, or scoring -- see harness.py's module docstring for
details.
"""
