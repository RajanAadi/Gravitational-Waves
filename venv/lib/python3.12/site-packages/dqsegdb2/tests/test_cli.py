# Copyright (C) 2024-2025 Cardiff University
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for `dqsegdb2.cli`.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import json

import pytest
from click.testing import CliRunner

from dqsegdb2 import cli


def _run(cmd, *args):
    """Run ``cmd`` with arguments through :class:`click.testing.CliRunner`.
    """
    runner = CliRunner()
    return runner.invoke(cmd, args)


def test_help():
    """Test that ``dqsegdb2 --help`` returns appropriately.
    """
    result = _run(cli.cli, "--help")
    assert result.exit_code == 0
    assert result.output.startswith("Usage: dqsegdb2 [OPTIONS] COMMAND")


def test_query_help():
    """Test that ``dqsegdb2 query --help`` returns appropriately.
    """
    result = _run(cli.cli, "query", "--help")
    assert result.exit_code == 0
    assert result.output.startswith("Usage: dqsegdb2 query [OPTIONS]")


@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize("tofile", (False, True))
def test_query_mock(requests_mock, tmp_path, raw, tofile):
    """Test ``dqsegdb2 query`` with a mocked response.
    """
    # construct mock response
    response = {
        "known": [[1000000000, 1000000010]],
        "active": [[1000000000, 1000000005], [1000000005, 10000000010]],
    }
    include = ["active", "known"]
    if raw:
        include.append("metadata")
    requests_mock.get(
        "https://segments.example.com/dq/X1/TEST/1"
        f"?s=1000000000.0&e=1000000010.0&include={'%2C'.join(include)}",
        json=response,
    )

    # run the command
    args = [
        "X1:TEST:1",
        "1000000000",
        "1000000010",
        "--server", "https://segments.example.com",
    ]
    if raw:
        args.append("--raw")
    if tofile:
        args.extend(("--output", str(tmp_path / "test.json")))
    result = _run(cli.cli, "query", *args)
    assert result.exit_code == 0

    # parse the JSON and check
    if tofile:
        with open(tmp_path / "test.json") as file:
            data = json.load(file)
    else:
        data = json.loads(result.output)
    assert data["known"] == response["known"]
    if raw:
        assert data["active"] == response["active"]
    else:  # coalesced
        assert data["active"] == [[1000000000, 1000000010]]
