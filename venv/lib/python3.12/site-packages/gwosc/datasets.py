# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT

"""
`gwosc.datasets` includes functions to query for available datasets.

To search for all available datasets:

>>> from gwosc import datasets
>>> datasets.find_datasets()
['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170814', 'GW170817', 'LVT151012', 'O1', 'S5', 'S6']
>>> datasets.find_datasets(detector="V1")
['GW170814', 'GW170817']
>>> datasets.find_datasets(type="run")
['O1', 'S5', 'S6']

To query for the GPS time of an event dataset (or vice-versa):

>>> datasets.event_gps("GW170817")
1187008882.43
>>> datasets.event_at_gps(1187008882)
'GW170817'

Similar queries are available for observing run datasets:

>>> datasets.run_segment("O1")
(1126051217, 1137254417)
>>> datasets.run_at_gps(1135136350)  # event_gps('GW151226')
'O1'

To run an event query filtered by merger parameters:

>>> from gwosc.datasets import query_events
>>> query_events(select=["mass-1-source <= 3.0"])
['GW170817-v3', 'GW190425-v1', 'GW190425-v2', 'GW190425_081805-v3']
"""  # noqa: E501

import re
import warnings

from . import (
    api,
    urls,
    utils,
)
from .api import DEFAULT_URL
from .api import v2 as apiv2

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_IGNORE = {
    "tenyear",
    "history",
    "oldhistory",
}


def _match_dataset(targetdetector, detectors, targetsegment, segment):
    """Returns `True` if the dataset metadata matches the target"""
    warnings.warn(
        "_match_dataset function will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    if targetdetector not in set(detectors) | {None}:
        return False
    if targetsegment is None or utils.segments_overlap(targetsegment, segment):
        return True


def _run_datasets(
    detector=None, segment=None, host=DEFAULT_URL, session=None, pagesize=None
):
    runs = apiv2.fetch_runs(host=host, session=session, pagesize=pagesize)
    for run in runs:
        # Skip if detector was given and it doesn't match
        if detector is not None and detector not in run["detectors"]:
            continue
        # Skip if segment was given and it doesn't overlap
        if segment is not None:
            run_segment = (run["gps_start"], run["gps_end"])
            do_overlap = utils.segments_overlap(run_segment, segment)
            if not do_overlap:
                continue
        yield run["name"]


def _catalog_datasets(host=DEFAULT_URL, session=None, pagesize=None):
    for catalog in apiv2.fetch_catalogs(host=host, session=session, pagesize=pagesize):
        yield catalog["name"]


def _match_event_dataset(
    dataset,
    catalog=None,
    version=None,
    detector=None,
    segment=None,
    host=DEFAULT_URL,
):
    warnings.warn(
        "_match_event_dataset function will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    # get strain file list (matching catalog and version)
    full = bool(detector or segment)
    try:
        meta = _event_metadata(
            dataset,
            catalog=catalog,
            version=version,
            full=full,
            host=host,
        )
    except ValueError:  # no dataset matching catalog and/or version
        return False

    if not full:  # detector=None, segment=None
        return True
    try:
        strain = meta["strain"]
    except KeyError:  # pragma: no cover
        # no strain file list for this dataset
        return False

    # match detector
    if detector not in {None} | {u["detector"] for u in strain}:
        return False

    # match segment
    if segment is None:
        return True
    if not strain:
        return False
    eseg = utils.strain_extent(urls.sieve(strain, detector=detector))
    return utils.segments_overlap(segment, eseg)


def _event_datasets(
    detector=None,
    segment=None,
    catalog=None,
    version=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    # This segment expansion is needed to account for segments
    # that do not contain events, but are contained in the
    # 4096s event dataset (strain file)
    if segment is not None:
        start, end = segment
        expanded_segment = (start, end + 4096)
    else:
        expanded_segment = None

    events = apiv2.fetch_event_versions(
        segment=expanded_segment,
        catalogs=catalog,
        host=host,
        session=session,
        pagesize=pagesize,
    )
    if version is not None:
        # Filter by version
        events = filter(lambda e: e["version"] == version, events)
    if detector is not None:
        # Filter by detector
        events = filter(lambda e: detector in e["detectors"], events)

    def _rank_catalog(cat):
        if "confident" in cat:
            return 1
        for word in ("marginal", "preliminary"):
            if word in cat:
                return 10
        return 5

    def _gps_distance(gps):
        if segment is None:
            return 0
        return int(abs(segment[0] - gps))

    for event in sorted(
        events,
        key=lambda e: (
            _gps_distance(e["gps"]),
            _rank_catalog(e["catalog"].lower()),
            -e["version"],
        ),
    ):
        yield f"{event['shortName']}"


def _iter_datasets(
    detector=None,
    type=None,
    segment=None,
    catalog=None,
    version=None,
    match=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    # get queries
    type = str(type).rstrip("s").lower()
    needruns = type in {"none", "run"}
    needcatalogs = type in {"none", "catalog"}
    needevents = type in {"none", "event"}

    # warn about event-only keywords when not querying for events
    if not needevents:
        for key, value in dict(catalog=catalog, version=version).items():
            if value is not None:
                warnings.warn(
                    f"the '{key}' keyword is only relevant when querying "
                    "for event datasets, it will be ignored now",
                )

    if match:
        reg = re.compile(match)

    def _yield_matches(iter_):
        for x in iter_:
            if not match or reg.search(x):
                yield x

    # search for events and datasets
    for needthis, collection in (
        (
            needruns,
            _run_datasets(
                detector=detector,
                host=host,
                segment=segment,
                session=session,
                pagesize=pagesize,
            ),
        ),
        (
            needcatalogs,
            _catalog_datasets(
                host=host,
                session=session,
                pagesize=pagesize,
            ),
        ),
        (
            needevents,
            _event_datasets(
                detector=detector,
                segment=segment,
                host=host,
                version=version,
                catalog=catalog,
                session=session,
                pagesize=pagesize,
            ),
        ),
    ):
        if not needthis:
            continue
        yield from _yield_matches(collection)


def find_datasets(
    detector=None,
    type=None,
    segment=None,
    match=None,
    catalog=None,
    version=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    """Find datasets available on the given GW open science host

    Parameters
    ----------
    detector : `str`, optional
        prefix of GW detector

    type : `str`, optional
        type of datasets to restrict, one of ``'run'``, ``'event'``, or
        ``'catalog'``

    segment : 2-`tuple` of `int`, `None`, optional
        a GPS ``[start, stop)`` interval to restrict matches to;
        datasets will match if they overlap at any point; this is
        not used to filter catalogs

    match : `str`, optional
        regular expression string against which to match datasets

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        the session to use for HTTP requests

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    datasets : `list` of `str`
        the names of all matched datasets, possibly empty

    Examples
    --------
    (Correct as of 2018-03-14)

    >>> from gwosc.datasets import find_datasets
    >>> find_datasets()
    ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170814', 'GW170817',
     'LVT151012', 'O1', 'S5', 'S6']
    >>> find_datasets(detector="V1")
    ['GW170814', 'GW170817']
    >>> find_datasets(type="event")
    ['GW150914', 'GW151226', 'GW170104', 'GW170608', 'GW170814', 'GW170817',
     'LVT151012']
    """
    return sorted(
        list(
            _iter_datasets(
                detector=detector,
                type=type,
                segment=segment,
                catalog=catalog,
                version=version,
                match=match,
                host=host,
                session=session,
                pagesize=pagesize,
            )
        )
    )


# -- event utilities ----------------------------------------------------------


def _event_metadata(
    event,
    catalog=None,
    version=None,
    full=True,
    host=DEFAULT_URL,
):
    warnings.warn(
        "_event_metadata function will be removed in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list(
        api._fetch_allevents_event_json(
            event,
            catalog=catalog,
            version=version,
            full=full,
            host=host,
        )["events"].values()
    )[0]


def event_gps(event, catalog=None, version=None, host=DEFAULT_URL, session=None):
    """Returns the GPS time of an open-data event

    Parameters
    ----------
    event : `str`
        the name of the event to query

    catalog : `str`, optional
        name of catalogue that hosts this event

    version : `int`, `None`, optional
        the version of the data release to use,
        defaults to the highest available version

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    Returns
    -------
    gps : `float`
        the GPS time of this event

    Examples
    --------
    >>> from gwosc.datasets import event_gps
    >>> event_gps("GW170817")
    1187008882.43
    >>> event_gps("GW123456")
    ValueError: no event dataset found for 'GW123456'
    """
    event = apiv2.fetch_event_version(
        event, catalog=catalog, version=version, host=host, session=session
    )
    return event["gps"]


def event_segment(
    event,
    detector=None,
    catalog=None,
    version=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    """Returns the GPS ``[start, stop)`` interval covered by an event dataset

    Parameters
    ----------
    event : `str`
        the name of the event

    detector : `str`, optional
        prefix of GW detector

    catalog : `str`, optional
        name of catalogue that hosts this event

    version : `int`, `None`, optional
        the version of the data release to use,
        defaults to the highest available version

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    pagesize : `int`, optional
        the number of results per page

    Returns
    -------
    start, end : `int`
        the GPS ``[start, end)`` interval covered by this event dataset

    Examples
    --------
    >>> from gwosc.datasets import event_segment
    >>> event_segment("GW150914")
    segment(1126257415, 1126261511)
    """

    strain_files = apiv2.fetch_event_strain_data(
        event,
        detector=detector,
        version=None,
        catalog=None,
        host=host,
        session=session,
        pagesize=pagesize,
    )

    if not strain_files:  # pragma: no cover
        raise ValueError(
            f"event '{event}' has no strain files",
        )

    # Return the strain files extent (mod of utils.strain_extent)
    starts, ends = zip(
        *[
            (afile["gps_start"], afile["gps_start"] + afile["duration"])
            for afile in strain_files
        ]
    )

    return min(starts), max(ends)


def event_at_gps(gps, host=DEFAULT_URL, tol=1, session=None, pagesize=None):
    """Returns the name of the open-data event matching the GPS time

    This function will return the first event for which
    ``|eventgps - gps| < = tol``.

    Parameters
    ----------
    gps : `float`
        The GPS time to locate

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    tol : `float`, optional
        the search window (in seconds), default: ``1``

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    pagesize : `int`, optional
        the number of results per page

    Returns
    -------
    event : `str`
        the name of the matched event

    Raises
    -------
    ValueError
        if no events are found matching the GPS time

    Examples
    --------
    >>> from gwosc.datasets import event_at_gps
    >>> event_at_gps(1187008882)
    'GW170817'
    >>> event_at_gps(1187008882, tol=0.1)
    ValueError: no event found within 0.1 seconds of 1187008882
    """
    events = list(
        apiv2.fetch_event_versions(
            segment=(gps - tol, gps + tol), session=session, pagesize=pagesize
        )
    )
    if len(events) < 1:
        raise ValueError(f"no event found within {tol} seconds of {gps}")
    return events[0]["name"]


def event_detectors(event, catalog=None, version=None, host=DEFAULT_URL, session=None):
    """Returns the `set` of detectors that observed an event

    Parameters
    ----------
    event : `str`
        the name of the event to query

    version : `int`, `None`, optional
        the version of the data release to use,
        defaults to the highest available version

    catalog : `str`, optional
        name of catalogue that hosts this event

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    Returns
    -------
    detectors : `set`
        the set of detectors for which data file URLs are included in
        the data release

    Examples
    --------
    >>> from gwosc.datasets import event_detectors
    >>> event_detectors("GW150914")
    {'H1', 'L1'}
    """
    event = apiv2.fetch_event_version(
        event, catalog=catalog, version=version, host=host, session=session
    )
    return set(event["detectors"])


# -- run utilities ------------------------------------------------------------


def run_segment(run, host=DEFAULT_URL, session=None):
    """Returns the GPS ``[start, stop)`` interval covered by a run dataset

    Parameters
    ----------
    run : `str`
        the name of the run dataset to query

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    Returns
    -------
    start, end : `int`
        the GPS ``[start, end)`` interval covered by this run dataset

    Examples
    --------
    >>> from gwosc.datasets import run_segment
    >>> run_segment("O1")
    segment(1126051217, 1137254417)
    >>> run_segment("Oh dear")
    ValueError: Run 'Oh dear' not found.
    """
    run = apiv2.fetch_run(run, session=session)
    return run["gps_start"], run["gps_end"]


def run_at_gps(gps, host=DEFAULT_URL, session=None, pagesize=None):
    """Returns the name of the open-data run dataset matching the GPS time

    This function will return the first event for which
    ``start <= gps < end``

    Parameters
    ----------
    gps : `float`
        The GPS time to locate

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    pagesize : `int`, optional
        the number of results per page

    Returns
    -------
    run : `str`
        the name of the matched observing run

    Raises
    -------
    ValueError
        if no datasets are found matching the GPS time

    Examples
    --------
    >>> from gwosc.datasets import run_at_gps
    >>> run_at_gps(1135136350)
    'O1'
    >>> run_at_gps(0)
    ValueError: no run dataset found containing GPS 0
    """
    for run in apiv2.fetch_runs(host=host, session=session, pagesize=pagesize):
        start, end = run["gps_start"], run["gps_end"]
        if start <= gps < end:
            return run["name"]
    raise ValueError(f"no run dataset found containing GPS {gps}")


def dataset_type(dataset, host=DEFAULT_URL, session=None, pagesize=None):
    """Returns the type of the named dataset

    Parameters
    ----------
    dataset : `str`
        The name of the dataset to match

    host : `str`, optional
        the URL of the GWOSC host to query

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    pagesize : `int`, optional
        the number of results per page

    Returns
    -------
    type : `str`
        the type of the dataset, one of 'run', 'event', or 'catalog'

    Raises
    -------
    ValueError
        if this dataset cannot be matched

    Examples
    --------
    >>> from gwosc.datasets import dataset_type
    >>> dataset_type("O1")
    'run'
    """
    for type_ in ("run", "catalog", "event"):
        if dataset in find_datasets(
            type=type_, host=host, session=session, pagesize=pagesize
        ):
            return type_
    raise ValueError(
        f"failed to determine type for dataset {dataset!r}",
    )


def query_events(select, host=DEFAULT_URL, session=None, pagesize=None):
    """Return a list of events filtered by the parameters in `select`

    Parameters
    ----------
    select : `list` of `str`
        A list of strings where each element is a range constrain on the
        event parameters.
        All ranges should have inclusive ends (<= and => operators).

    host : `str`, optional
        the URL of the GWOSC host to query

    session : `requests.Session`, optional
        HTTP session to use for making requests, defaults to
        using a new session for each API call

    pagesize : `int`, optional
        the number of results per page

    Examples
    --------
    >>> from gwosc.datasets import query_events
    >>> query_events(
    ...     select=[
    ...         "mass-1-source >= 1.4",
    ...         "200 >= luminosity-distance >= 100",
    ...     ]
    ... )
    ['GW190425-v1', 'GW190425-v2', 'GW190425_081805-v3']

    Notes
    -----
    Operators:

    - `<=` (or `=<`)
    - `=>` (or `>=`)

    Parameters:

    - ``gps-time`` [s],
    - ``mass-1-source`` [solar mass],
    - ``mass-2-source`` [solar mass],
    - ``network-matched-filter-snr``,
    - ``luminosity-distance`` [Mpc],
    - ``chi-eff``,
    - ``total-mass-source`` [solar mass],
    - ``chirp-mass`` [solar mass],
    - ``chirp-mass-source`` [solar mass],
    - ``redshift``,
    - ``far`` [events/year],
    - ``p-astro``,
    - ``final-mass-source`` [solar mass],

    For a full description of all parameters see
    https://www.gwosc.org/apidocs/#event5
    """
    select_query = dict(utils.select_to_query(select, host=host))
    events = apiv2.fetch_event_versions(
        select=select_query, host=host, session=session, pagesize=pagesize
    )
    return [f"{e['shortName']}" for e in events]
