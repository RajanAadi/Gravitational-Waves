# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT
import logging
from urllib.parse import urlencode

import requests

from gwosc import __version__

from .. import utils
from . import DEFAULT_URL

__all__ = [
    "fetch_json",
    "fetch_dataset_json",
    "fetch_run_json",
    "fetch_allowed_params_json",
    "fetch_cataloglist_json",
    "fetch_catalog_json",
    "fetch_allevents_json",
    "fetch_filtered_events_json",
    "fetch_event_json",
]

logger = logging.getLogger(__name__)

_MAX_GPS = 99999999999

#: Cache of downloaded blobs
JSON_CACHE = {}

# Timeout before a request is dropped
TIMEOUT = 10  # seconds

# -- JSON handling ------------------------------------------------------------


def fetch_json(url, **kwargs):
    """Fetch JSON data from a remote URL

    Parameters
    ----------
    url : `str`
        the remote URL to fetch

    kwargs : dict
        other keyword arguments are passed directly to :func:`requests.get`

    Returns
    ------
    data : `dict` or `list`
        the data fetched from ``url`` as parsed by
        :meth:`requests.Response.json`

    See also
    --------
    json.loads
        for details of the JSON parsing
    """
    try:
        return JSON_CACHE[url]
    except KeyError:
        logger.debug("fetching %s", url)
        client_headers = {"User-Agent": f"python-gwosc/{__version__}"}
        resp = requests.get(url, headers=client_headers, timeout=TIMEOUT, **kwargs)
        resp.raise_for_status()
        return JSON_CACHE.setdefault(
            url,
            resp.json(),
        )


# -- Run datasets -------------------------------------------------------------


def _dataset_url(start, end, host=DEFAULT_URL):
    return f"{host}/archive/{start:d}/{end:d}/json/"


def fetch_dataset_json(gpsstart, gpsend, host=DEFAULT_URL):
    """Returns the JSON metadata for all datasets matching the GPS interval

    Parameters
    ----------
    gpsstart : `int`
        the GPS start of the desired interval

    gpsend : `int`
        the GPS end of the desired interval

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returned by `json.loads`
    """
    return fetch_json(_dataset_url(gpsstart, gpsend, host=host))


def _run_url(run, detector, start, end, host=DEFAULT_URL):
    return f"{host}/archive/links/{run}/{detector}/{start:d}/{end:d}/json/"


def fetch_run_json(run, detector, gpsstart=0, gpsend=_MAX_GPS, host=DEFAULT_URL):
    """Returns the JSON metadata for the given science run parameters

    Parameters
    ----------
    run : `str`
        the name of the science run, e.g. ``'O1'``

    detector : `str`
        the prefix of the GW detector, e.g. ``'L1'``

    gpsstart : `int`
        the GPS start of the desired interval

    gpsend : `int`
        the GPS end of the desired interval

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returned by `json.loads`
    """
    return fetch_json(_run_url(run, detector, gpsstart, gpsend, host=host))


# -- EventAPI catalogs -------------------------------------------------------


def _allowed_params_url(host=DEFAULT_URL):
    return f"{host}/eventapi/json/query/params/"


def fetch_allowed_params_json(host=DEFAULT_URL):
    return fetch_json(_allowed_params_url(host=host))


def _eventapi_url(full=False, host=DEFAULT_URL):
    j = "jsonfull" if full else "json"
    return f"{host}/eventapi/{j}/"


def fetch_cataloglist_json(host=DEFAULT_URL):
    """Returns the JSON metadata for the catalogue list.

    Parameters
    ----------
    host : `str`, optional
        the URL of the GWOSC host to query

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returned by
        :meth:`requests.Response.json`
    """
    return fetch_json(_eventapi_url(host=host))


def _catalog_url(catalog, host=DEFAULT_URL):
    return f"{_eventapi_url(host=host)}{catalog}/"


def fetch_catalog_json(catalog, host=DEFAULT_URL):
    """Returns the JSON metadata for the given catalogue

    Parameters
    ----------
    catalog : `str`
        the name of the event catalog, e.g. `GWTC-1-confident`

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returnend by
        :meth:`requests.Response.json`
    """
    return fetch_json(_catalog_url(catalog, host=host))


# -- EventAPI events ---------------------------------------------------------


def _allevents_url(full=False, host=DEFAULT_URL):
    return f"{_eventapi_url(full=full, host=host)}allevents/"


def _has_jsonfull_allevents(host=DEFAULT_URL):
    return _allevents_url(full=True, host=host) in JSON_CACHE


def fetch_allevents_json(full=False, host=DEFAULT_URL):
    """Returns the JSON metadata for the allevents API

    Parameters
    ----------
    host : `str`, optional
        the URL of the GWOSC host to query, defaults to https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returned by
        :meth:`requests.Response.json`
    """
    if full is None and _has_jsonfull_allevents(host=host):
        return fetch_json(_allevents_url(full=True, host=host))
    return fetch_json(_allevents_url(full=full, host=host))


def _fetch_allevents_event_json(
    event,
    catalog=None,
    version=None,
    full=False,
    host=DEFAULT_URL,
):
    """Returns the JSON metadata from the allevents view for a specific event

    The raw JSON data are packaged to look the same as if they came from
    a full event API query, i.e. nested under `'events`'.
    """
    allevents = fetch_allevents_json(full=full, host=host)["events"]
    matched = []

    def _match(keyvalue):
        dset, metadata = keyvalue
        if event not in {
            dset,
            metadata["commonName"],  # full name
            metadata["commonName"].split("_", 1)[0],  # GWYYMMDD prefix
        }:
            return
        if version is not None and metadata["version"] != version:
            return
        if catalog is not None and metadata["catalog.shortName"] != catalog:
            return
        return True

    # match datasets
    matched = list(filter(_match, allevents.items()))
    names = {x[1]["commonName"] for x in matched}

    # one dataset has an exact name match, so discard everything else
    if event in names:
        matched = [x for x in matched if x[1]["commonName"] == event]
        names = {x[1]["commonName"] for x in matched}

    # we have a winner!
    if len(names) == 1:
        key, meta = sorted(matched, key=lambda x: x[1]["version"])[-1]
        return {"events": {key: meta}}

    # raise error with the right message
    if len(names) > 1:
        raise ValueError(
            "multiple events matched for {!r}: '{}'".format(
                event,
                "', '".join(names),
            ),
        )
    msg = "failed to identify {} for event '{}'"
    if catalog is None:
        msg = msg.format("catalog", event)
        if version is not None:
            msg += f" at version {version}"
        raise ValueError(msg)
    msg = msg.format("version", event)
    if catalog is not None:
        msg += f" in catalog '{catalog}'"
    raise ValueError(msg)


def _query_events_url(select, host=DEFAULT_URL):
    query = urlencode(utils.select_to_query(select, host=host))
    return f"{host}/eventapi/json/query/show?{query}"


def fetch_filtered_events_json(select, host=DEFAULT_URL):
    """Return the JSON metadata for the events constrained by select

    Parameters
    ----------
    select : `list-like`
        a list of range constrains for the events.
        All ranges should have inclusive ends (<= and >= operators).

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returnend by
        :meth:`requests.Response.json`

    Example
    -------
    >>> fetch_filtered_events_json(
    ...     select=[
    ...         "mass-1-source <= 5",
    ...         "mass-2-source =< 10",
    ...         "10 <= luminosity-distance <= 100",
    ...     ]
    ... )
    """
    return fetch_json(_query_events_url(select, host=host))


def _event_url(
    event,
    catalog=None,
    version=None,
    host=DEFAULT_URL,
):
    return list(
        _fetch_allevents_event_json(
            event,
            catalog=catalog,
            version=version,
            full=None,
            host=host,
        )["events"].values()
    )[0]["jsonurl"]


def fetch_event_json(
    event,
    catalog=None,
    version=None,
    host=DEFAULT_URL,
):
    """Returns the JSON metadata for the given event.

    By default, this function queries across all catalogs and all data-release
    versions, returning the highest available version, unless the
    ``version`` and/or ``catalog`` keywords are specified.

    Parameters
    ----------
    event : `str`
        the name of the event to query

    catalog : `str`, optional
        name of catalogue that hosts this event

    version : `int`, `None`, optional
        restrict query to a given data-release version

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    Returns
    -------
    data : `dict` or `list`
        the JSON data retrieved from GWOSC and returned by `json.loads`
    """
    return fetch_json(
        _event_url(event, catalog=catalog, version=version, host=host),
    )
