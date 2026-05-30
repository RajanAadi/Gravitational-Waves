# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT

import re
from os.path import basename

from .api import DEFAULT_URL
from .api.v2 import fetch_allowed_params

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_ALLOWED_OPS = {">=", "=>", "<=", "=<"}


def url_segment(url):
    """Return the GPS segment covered by a URL following T050017

    Parameters
    ---------
    filename : `str`
        the path name of a file

    Returns
    -------
    segment : `tuple` of `int`
        the ``[start, stop)`` GPS interval covered by the given URL

    Notes
    -----
    `LIGO-T050017 <https://dcc.ligo.org/LIGO-T050017/public>`_ declares
    a filenaming convention that includes documenting the GPS start integer
    and integer duration of a file, see that document for more details.
    """
    base = basename(url)
    _, _, s, e = base.split("-")
    s = int(s)
    e = int(e.split(".")[0])
    return s, s + e


def url_overlaps_segment(url, segment):
    """Returns True if a URL overlaps a [start, stop) GPS interval

    Parameters
    ----------
    url : `str`
        the URL of a file

    segment : `tuple` of `int`
        the ``[start, stop)`` GPS interval to check against

    Returns
    -------
    overlap? : `bool`
        `True` if the GPS interval for the URL matches that given,
        otherwise `False`
    """
    useg = url_segment(url)
    return segments_overlap(useg, segment)


def urllist_extent(urls):
    """Returns the GPS `[start, end)` interval covered by a list or URLs

    Parameters
    ----------
    urls : `iterable` of `str`
        the list of URLs

    Returns
    -------
    a, b : 2-`tuple` of int`
        the GPS extent of the URL list
    """
    segs = map(url_segment, urls)
    starts, ends = zip(*segs)
    return min(starts), max(ends)


def strain_extent(strain):
    """Returns the GPS `[start, end)` interval covered by a strain meta dict"""
    starts, ends = zip(
        *[(meta["GPSstart"], meta["GPSstart"] + meta["duration"]) for meta in strain]
    )
    return min(starts), max(ends)


def full_coverage(urls, segment):
    """Returns True if the list of URLS completely covers a GPS interval

    The URL list is presumed to be contiguous, so this just checks that
    the first URL (by GPS timestamp) and the last URL can form a segment
    that overlaps that given.
    """
    if not urls:
        return False
    # sort URLs by GPS timestamp
    a, b = urllist_extent(urls)
    return a <= segment[0] and b >= segment[1]


def segments_overlap(a, b):
    """Returns True if GPS segment ``a`` overlaps GPS segment ``b``"""
    return (a[1] > b[0]) and (a[0] < b[1])


def parse_two_ops(compiled_m, host=DEFAULT_URL):
    allowed_params = fetch_allowed_params(host=host)
    md = compiled_m.groupdict()
    op1, op2 = md["op1"], md["op2"]
    if not {op1, op2}.issubset(_ALLOWED_OPS):
        raise ValueError("Could not parse select string.\n" "Unknown operators.")
    param, val1, val2 = md["param"], md["val1"], md["val2"]
    if param not in allowed_params:
        raise ValueError(
            "Could not parse select string.\n"
            f"Unrecognized parameter: {param}.\n"
            f"Use one of:\n{allowed_params}"
        )
    queries = []
    if ">" in op1:
        queries.append((f"max-{param}", val1))
    if "<" in op1:
        queries.append((f"min-{param}", val1))
    if ">" in op2:
        queries.append((f"min-{param}", val2))
    if "<" in op2:
        queries.append((f"max-{param}", val2))
    return queries


def parse_one_op(compiled_m, host=DEFAULT_URL):
    allowed_params = fetch_allowed_params(host=host)
    md = compiled_m.groupdict()
    op = md["op"]
    if not {op}.issubset(_ALLOWED_OPS):
        raise ValueError("Could not parse select string.\n" "Unknown operator.")
    param, val = md["param"], md["val"]
    if param not in allowed_params:
        raise ValueError(
            f"Could not parse select string.\n"
            f"Unrecognized parameter: {param}.\n"
            f"Use one of:\n{allowed_params}"
        )
    queries = []
    if ">" in op:
        queries.append((f"min-{param}", val))
    if "<" in op:
        queries.append((f"max-{param}", val))
    return queries


def select_to_query(select, host=DEFAULT_URL):
    """Parse select string and translate into URL GET parameters"""

    # Captures strings of the form `1.44 <= param <= 5.0`
    two_ops = re.compile(
        r"^\s*(?P<val1>[\d.+-eE]+)\s*(?P<op1>[<>=]{2})\s*(?P<param>[\w-]+)"
        r"\s*(?P<op2>[<>=]+)\s*(?P<val2>[\d.+-eE]+)\s*$"
    )
    # Captures strings of the form `param <= 5.0`
    one_op = re.compile(
        r"^\s*(?P<param>[\w-]+)\s*(?P<op>[<>=]{2})\s*(?P<val>[\d.+-eE]+)\s*$"
    )

    queries = []
    for s in select:
        for regex, parse in (
            (one_op, parse_one_op),
            (two_ops, parse_two_ops),
        ):
            m = regex.match(s)
            if m is not None:
                queries.extend(parse(m, host=host))
                break
        else:
            raise ValueError(f"Could not parse select string: {s}")
    return queries
