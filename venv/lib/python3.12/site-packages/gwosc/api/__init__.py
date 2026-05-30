"""
:mod:`gwosc.api` provides the low-level interface functions
that handle direct requests to the GWOSC host.
"""

import os

#: The default GWOSC host URL
DEFAULT_URL = os.getenv("GWOSC_SERVER_URL", "https://gwosc.org")

from .v1 import *  # noqa

# Internal attributes for backwards compatibility
from .v1 import _MAX_GPS, _fetch_allevents_event_json  # noqa
