# Copyright (C) 2024-2025 Cardiff University
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command-line interface for DQSEGDB2.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import json

import click

from dqsegdb2 import query as dqsegdb2_query
from dqsegdb2.utils import get_default_host


@click.group("dqsegdb2")
def cli():
    """Command-line interface for DQSEGDB2.
    """


@cli.command()
@click.argument("flag")
@click.argument("gpsstart", type=float)
@click.argument("gpsend", type=float)
@click.option(
    "-s", "--server",
    default=get_default_host(),
    show_default=True,
    help="Address of DQSegDB server to talk to.",
)
@click.option(
    "-r", "--raw",
    is_flag=True,
    default=False,
    show_default=True,
    help="Write 'raw' JSON response from server.",
)
@click.option(
    "-o", "--output",
    help="Path in which to write output, default: stdout.",
)
def query(flag, gpsstart, gpsend, server, raw, output):
    """Query for a FLAG in the interval `[GPSSTART, GPSEND)`.

    Output is printed to the screen as JSON.
    """
    data = dqsegdb2_query.query_segments(
        flag,
        gpsstart,
        gpsend,
        raw=raw,
        host=server,
    )
    jdat = json.dumps(data, indent=None)
    if output is None:
        return print(jdat)
    with open(output, "w") as f:
        return f.write(jdat)
