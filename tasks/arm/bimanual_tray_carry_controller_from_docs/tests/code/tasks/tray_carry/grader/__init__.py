import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def published_table(frozen=None):
    import published
    return published.published_table(frozen)
