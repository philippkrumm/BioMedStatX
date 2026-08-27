"""Fuzzing harnesses for BioMedStatX.

``src`` is put on the path here rather than in each module. The workers already
did it for themselves, but the orchestrators import ``html_oracles`` in the
parent process just to read the oracle names for their coverage summary -- and
that module now imports the report checks from ``src/export/report_selfcheck``,
which is where they belong once the export path runs them too. Without this the
parent process could no longer import what its children could.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
