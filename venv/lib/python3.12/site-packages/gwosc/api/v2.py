# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT
import logging
import warnings
from contextlib import nullcontext
from urllib import parse

import requests
import tenacity

from gwosc import __version__

from . import DEFAULT_URL

logger = logging.getLogger(__name__)

CLIENT_HEADERS = {"User-Agent": f"python-gwosc/{__version__}"}

#: Cache of downloaded blobs
JSON_CACHE = {}

#: Number of pages to fetch that trigger a warning
LARGE_NUM_PAGES_WARN = 100

# Status codes on which we can retry
TRANSIENT_ERROR_CODES = [
    307,
    408,
    413,
    429,
    500,
    501,
    502,
    503,
    504,
]
# Number of request retries after a transient error, before we raise an exception
MAX_RETRIES = 10
# Default wait time before we retry, if the `Retry-After` header key in the
# response is not set, or if it can't be parsed as an integer
RETRY_DEFAULT_WAIT_TIME = 1  # seconds

# Timeout before a request is dropped
TIMEOUT = 10  # seconds

# -- Retry handling ------------------------------------------------------------
# We use tenacity to handle transient network/HTTP errors
# (see https://git.ligo.org/gwosc/client/-/merge_requests/121)


def _retry_on_transient_errors(retry_state):
    """This function is called by tenacity to determine if we should retry.
    It is called only when we received an HTTP response (`requests.Timeout` are
    processed elsewhere).

    We retry if the HTTP status code indicates a transient error.
    """
    exc = retry_state.outcome.exception()
    url = retry_state.args[1]
    if exc is not None:
        if isinstance(exc, requests.Timeout):
            # There was a Timeout
            logger.log(logging.INFO, f"Got timeout for {url}")
            return True
        else:
            # There was another exception
            return False
    else:
        # We got a response so let's inspect it
        resp = retry_state.outcome.result()
        if resp.status_code in TRANSIENT_ERROR_CODES:
            logger.log(logging.INFO, f"Got status code {resp.status_code} for {url}")
            return True
        else:
            return False


def _dynamic_wait(retry_state):
    """This function is used by tenacity to determine how long to wait.
    It is called whether there was a `requests.Timeout` or an actual HTTP response.

    If we have an HTTP response and the header contains a `Retry-After` field, use it.
    Else, wait for a fixed amount.
    """
    # retry_state.outcome is a Future that represents the result of _fetch_json.
    # We know that either there was a Timeout either we have an HTTP response.
    if retry_state.outcome.exception():
        # There was a Timeout exception
        retry_after = RETRY_DEFAULT_WAIT_TIME
    else:
        # We got a response so let's inspect it
        resp = retry_state.outcome.result()
        try:
            retry_after = int(resp.headers.get("Retry-After", RETRY_DEFAULT_WAIT_TIME))
        except ValueError:
            retry_after = RETRY_DEFAULT_WAIT_TIME
    # We can access the URL from the retry_state
    url = retry_state.args[1]
    logger.log(logging.INFO, f"Will retry {url} in {retry_after} seconds")
    return retry_after


@tenacity.retry(
    retry=_retry_on_transient_errors,
    wait=_dynamic_wait,
    stop=tenacity.stop_after_attempt(MAX_RETRIES),
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
)
def _fetch_json(get_func, url, **kwargs):
    """Low level function to fetch the data with retry.
    """
    return get_func(url, headers=CLIENT_HEADERS, timeout=TIMEOUT, **kwargs)


# -- JSON handling ------------------------------------------------------------


def fetch_json(url, session=None, **kwargs):
    """Fetch JSON data from a remote URL.

    Parameters
    ----------
    url : `str`
        the remote URL to fetch

    session : `requests.Session`, optional
        the session to use for the request. If None, falls back to
        direct requests.get()

    kwargs : `dict`
        other keyword arguments are passed directly to :func:`requests.get`
        or :meth:`requests.Session.get`

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
    if url in JSON_CACHE.keys():
        return JSON_CACHE[url]

    logger.debug("fetching %s", url)

    # Determine which session/method to use
    if session is not None:
        # Use provided session
        get_func = session.get
    else:
        # Fall back to direct requests.get
        get_func = requests.get

    # Fetch with retry
    try:
        resp = _fetch_json(get_func, url, **kwargs)
    except tenacity.RetryError:
        logger.log(logging.ERROR, f"Too much trials for {url}")
        raise
    resp.raise_for_status()
    return JSON_CACHE.setdefault(url, resp.json())


def produce_fetched_objects(url, session=None):
    if session is None:
        session = requests.Session()
    else:
        # don't close an existing session
        session = nullcontext(session)
    with session as sess:
        current_page = fetch_json(url, session=sess)
        num_pages = current_page["num_pages"]
        if num_pages > LARGE_NUM_PAGES_WARN:
            warn_message = (
                f"Your request will need to fetch for {num_pages} pages. "
                "Try to constraint your request if possible."
            )
            warnings.warn(warn_message)
            logger.warning(warn_message)
        yield from current_page["results"]
        # Iterate over the rest of the pages
        next_page = current_page["next"]
        while next_page is not None:
            current_page = fetch_json(next_page, session=sess)
            next_page = current_page["next"]
            yield from current_page["results"]


def fetch_run_strain_files(
    run=None,
    detector=None,
    start=None,
    end=None,
    sample_rate=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    """Return strain file objects from bulk-data releases.

    Parameters
    ----------
    run : `str`, optional
        the ID of a run, e.g. ``'O1'``.

    detector : `str`, optional
        the prefix of the GW detector, e.g. ``'L1'``.

    start : `int`, optional
        the GPS start of the desired interval.

    end : `int`, optional
        the GPS end of the desired interval.

    sample_rate : `int`, optional
        the sample rate of the strain file data, either 4096 or 16384 [Hz].

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        An iterable of strain file dictionaries.
    """

    if run is not None:
        if "KHZ_R1" in run:
            run = run.split("_")[0]
        strain_url = f"{host}/api/v2/runs/{run}/strain-files"
    else:
        strain_url = f"{host}/api/v2/strain-files"

    # Prepare query parameters
    start = None if start is None else int(start)
    end = None if end is None else int(end)
    sample_rate = {4096: 4, 16_384: 16}.get(sample_rate, sample_rate)
    query_params = {
        k: v
        for k, v in {
            "detector": detector,
            "start": start,
            "stop": end,
            "sample-rate": sample_rate,
            "pagesize": pagesize,
        }.items()
        if v is not None
    }
    query_string = parse.urlencode(query_params)

    if query_string:
        strain_url = f"{strain_url}?{query_string}"

    yield from produce_fetched_objects(strain_url, session=session)


def fetch_event_strain_data(
    event,
    detector=None,
    version=None,
    catalog=None,
    sample_rate=None,
    duration=None,
    format="hdf5",
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    """Return strain file objects from single-event releases.

    Parameters
    ----------
    event : `str` or `None`
        the ID of an event, e.g. ``'GW150914'``.

    detector : `str`
        the prefix of the GW detector, e.g. ``'L1'``.

    version : `int`
        the version number of the requested event.

    catalog : `str`
        the catalog in which the requested event appears.

    sample_rate : `int`
        the sample rate of the strain file data, either 4096 or 16384 [Hz].

    duration : `int`
        the duration of the strain file, 32 or 4096 [s].

    format : `str`
        the file format of the strain file.
        One of ``'hdf'``, ``'gwf'``, ``'txt'``.

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        An iterable of strain file dictionaries.
    """

    # Prepare event ID
    event_id = event
    if "-v" not in event:
        if version is not None:
            event_id = f"{event}-v{version}"
        elif catalog is not None:
            event_id = f"{event}@{catalog}"

    strain_url = f"{host}/api/v2/event-versions/{event_id}/strain-files"

    # Prepare query parameters
    format = "hdf" if format == "hdf5" else format
    sample_rate = {4096: 4, 16_384: 16}.get(sample_rate, sample_rate)
    query_params = {
        k: v
        for k, v in {
            "detector": detector,
            "sample-rate": sample_rate,
            "file-format": format,
            "duration": duration,
            "pagesize": pagesize,
        }.items()
        if v is not None
    }
    query_string = parse.urlencode(query_params)

    if query_string:
        strain_url = f"{strain_url}?{query_string}"

    yield from produce_fetched_objects(strain_url, session=session)


def fetch_segments(flag, start, end, host=DEFAULT_URL, session=None, pagesize=None):
    """Return segment dictionaries in the (start, end) GPS time interval.

    Parameters
    ----------
    flag : `str`
        name of flag, e.g. ``'H1_DATA'``.

    start : `int`
        the GPS start time of your query.

    end : `int`
        the GPS end time of your query.

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        An iterable of segment dictionaries.
    """
    segments_url = f"{host}/api/v2/timelines/{flag}/segments?start={start}&stop={end}"
    if pagesize is not None:
        segments_url = f"{segments_url}&pagesize={pagesize}"
    yield from produce_fetched_objects(segments_url, session=session)


def fetch_runs(host=DEFAULT_URL, session=None, pagesize=None):
    """Return a list of all past runs.

    Parameters
    ----------
    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        An iterable of runs.
    """
    url = f"{host}/api/v2/runs"
    if pagesize is not None:
        url = f"{url}?pagesize={pagesize}"
    yield from produce_fetched_objects(url, session=session)


def fetch_catalogs(host=DEFAULT_URL, session=None, pagesize=None):
    """Returns a list with all catalogs.

    Parameters
    ----------
    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        A list of catalogs.
    """
    url = f"{host}/api/v2/catalogs"
    if pagesize is not None:
        url = f"{url}?pagesize={pagesize}"
    yield from produce_fetched_objects(url, session=session)


def fetch_event_versions(
    name=None,
    segment=None,
    catalogs=None,
    select=None,
    host=DEFAULT_URL,
    session=None,
    pagesize=None,
):
    """Returns an event.

    Parameters
    ----------
    name : `str`, optional
        a full or partial name for an event.

    segment : `tuple`, optional
        a gps time tuple (start, end) to restrict the search.

    catalogs : `str`, `iterable`, optional
        a single catalog name or a list of catalog names.

    select : `dict`, optional
        a dictionary with query parameters, e.g. {'min-p-astro': 0.5}.

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    pagesize : `int`, optional
        the number of results per page.

    Returns
    -------
    data : `iterable[dict]`
        A list of event version dictionaries.
    """
    events_url = f"{host}/api/v2/event-versions"

    query = {}
    if name is not None:
        query["name-contains"] = name

    if segment is not None:
        query["min-gps-time"], query["max-gps-time"] = segment

    if catalogs is not None:
        if isinstance(catalogs, str):
            query["release"] = catalogs
        else:
            if len(catalogs) > 0:
                query["release"] = ",".join(catalogs)

    if select is not None:
        query = {**query, **select}

    if pagesize is not None:
        query["pagesize"] = pagesize

    query_string = parse.urlencode(query)

    if query_string:
        events_url = f"{events_url}?{query_string}"

    yield from produce_fetched_objects(events_url, session=session)


def fetch_run(run, host=DEFAULT_URL, session=None):
    """Return a run detail.

    Parameters
    ----------
    run : `str`
        the name of the run, e.g. O1.

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    Returns
    -------
    data : `dict`
        A dictionary with the run detail.
    """
    try:
        run = fetch_json(f"{host}/api/v2/runs/{run}", session=session)
    except requests.HTTPError:
        raise ValueError(f"Run '{run}' not found.")
    return run


def fetch_event_version(
    event, catalog=None, version=None, host=DEFAULT_URL, session=None
):
    """Returns an event.

    Parameters
    ----------
    event : `str`
        the name of the event.

    catalog : `str`, optional
        name of catalogue that hosts this event.

    version : `int`, `None`, optional
        the version of the data release to use,
        defaults to the highest available version.

    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    Returns
    -------
    data : `dict`
        A dictionary with the event detail.
    """
    event_string = event
    if "-v" in event:
        event_string = event
    elif version is not None:
        event_string = f"{event}-v{version}"
    elif catalog is not None:
        event_string = f"{event}@{catalog}"
    event_url = f"{host}/api/v2/event-versions/{event_string}"

    return fetch_json(event_url, session=session)


def fetch_allowed_params(host=DEFAULT_URL, session=None):
    """Return a list with the "default parameters".

    These parameters are almost always estimated for event detections.
    Use the parameter names (strings) to filter events in the
    :func:`fetch_event_versions` `select` argument.

    Parameters
    ----------
    host : `str`, optional
        the URL of the GWOSC host to query, defaults to
        https://gwosc.org.

    session : `requests.Session`, optional
        the session to use for HTTP requests.

    Returns
    -------
    params : `list[str]`
        A list of parameter names.
    """
    return list(
        produce_fetched_objects(f"{host}/api/v2/default-parameters", session=session)
    )
