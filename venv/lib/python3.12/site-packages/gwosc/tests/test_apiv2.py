# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT

"""Tests for :mod:`gwosc.api.v2`"""

from unittest.mock import patch

import pytest
import requests
import tenacity

from ..api import v2 as apiv2

__author__ = "Martin Beroiz <martin.beroiz@ligo.org>"

# Those values are used to mock requests
# This is used in tests of the retry mechanism.
# Note that fetch_json uses cache so each test has to use a different mock URL
# so we use the name of the test.
# Also as those tests don't check exactly how much time we wait,
# we patch RETRY_DEFAULT_WAIT_TIME to 0.
# TODO: The tests are a bit redundant so maybe we can refactor them.
response_429 = {
    "status_code": 429,
    "headers": {
        "Retry-After": "0"  # We use 0 to speed-up tests
    },
}

response_200 = {"status_code": 200, "text": "[1, 2, 3]"}

response_500 = {
    "status_code": 500,
}

response_404 = {
    "status_code": 404,
}

exception_timeout = {"exc": requests.ReadTimeout}


def test_cache():
    apiv2.JSON_CACHE["something"] = "cached_value"
    assert apiv2.fetch_json("something") == "cached_value"


@patch("gwosc.api.v2.fetch_json")
def test_produce_fetched_objects_pagination(mock_fetch):
    mock_fetch.side_effect = [
        {
            "next": "http://dummy_url.com?page=2",
            "results": [1, 2, 3],
            "num_pages": 10,
        },
        {
            "next": None,
            "results": [4, 5, 6],
            "num_pages": 10,
        },
    ]
    result = list(apiv2.produce_fetched_objects("mock_url"))
    assert result == [1, 2, 3, 4, 5, 6]


@patch("gwosc.api.v2.fetch_json")
def test_produce_fetched_objects_toomanypages(mock_fetch):
    mock_fetch.side_effect = [
        {
            "next": "http://dummy_url.com?page=2",
            "results": [1, 2, 3],
            "num_pages": 1000,
        },
        {
            "next": None,
            "results": [4, 5, 6],
            "num_pages": 1000,
        },
    ]
    with pytest.warns(UserWarning, match="1000 pages"):
        result = list(apiv2.produce_fetched_objects("mock_url"))
        assert result == [1, 2, 3, 4, 5, 6]


@patch("gwosc.api.v2.produce_fetched_objects")
def test_pagesize_is_passed(mock_produce):
    """Test that pagesize parameter is correctly passed to the URL."""
    mock_produce.return_value = iter([])

    # Test cases with required parameters for each function
    test_cases = [
        (apiv2.fetch_event_versions, {}),
        (apiv2.fetch_runs, {}),
        (apiv2.fetch_catalogs, {}),
        (
            apiv2.fetch_segments,
            {"flag": "H1_DATA", "start": 932540000, "end": 932560000},
        ),
        (apiv2.fetch_run_strain_files, {"run": "S5"}),
        (apiv2.fetch_event_strain_data, {"event": "GW150914"}),
    ]

    for func, required_params in test_cases:
        # Reset the mock
        mock_produce.reset_mock()

        # Call function and consume the generator,
        # so that produce_fetched_objects is called
        list(func(**required_params, pagesize=23))

        # Verify that produce_fetched_objects was called with correct url
        mock_produce.assert_called_once()
        call_args = mock_produce.call_args
        url_arg = call_args[0][0]
        assert "pagesize=23" in url_arg


@pytest.mark.remote
def test_fetch_run_strain_files():
    s5_files = list(
        apiv2.fetch_run_strain_files(
            detector="H1", run="S5", start=826175488, end=826200064
        )
    )
    assert len(s5_files) == 4
    o1_files = list(
        apiv2.fetch_run_strain_files(
            detector="L1", run="O1", start=1127407616, end=1127420000
        )
    )
    assert len(o1_files) == 4


@pytest.mark.remote
def test_fetch_event_strain_data():
    gw_files = list(
        apiv2.fetch_event_strain_data(
            "GW150914", version=3, sample_rate=4096, format="txt"
        )
    )
    assert len(gw_files) == 4
    for afile in gw_files:
        assert afile["sample_rate_kHz"] == 4
        assert afile["file_format"].lower() == "txt"


@pytest.mark.remote
def test_fetch_segments():
    segs = list(apiv2.fetch_segments("H1_DATA", start=932540000, end=932560000))
    assert len(segs) == 2


@pytest.mark.remote
def test_fetch_runs():
    assert "S6" in {run["name"] for run in apiv2.fetch_runs()}


@pytest.mark.remote
def test_fetch_catalogs():
    catalogs = {cat["name"] for cat in apiv2.fetch_catalogs()}
    assert "GWTC" in catalogs
    assert "GWTC-1-confident" in catalogs


@pytest.mark.remote
def test_fetch_event_versions():
    event = list(apiv2.fetch_event_versions(name="GW150914"))[0]
    assert event["name"] == "GW150914"

    events = list(apiv2.fetch_event_versions(catalogs="GWTC-1-confident"))
    assert len(events) == 11

    event = list(apiv2.fetch_event_versions(segment=(1126259450, 1126259470)))[0]
    assert event["name"] == "GW150914"

    events = list(
        apiv2.fetch_event_versions(
            select={
                "max-mass-1-source": 5,
                "min-p-astro": 0.5,
            }
        )
    )[0]
    assert len(events) > 0


@pytest.mark.remote
def test_fetch_run():
    run = apiv2.fetch_run("S5")
    assert run["name"] == "S5"


@pytest.mark.remote
def test_fetch_event_version():
    event = apiv2.fetch_event_version("GW150914", version=3)
    assert "150914" in event["name"]
    assert event["version"] == 3

    event = apiv2.fetch_event_version("GW150914", catalog="GWTC")
    assert event
    assert "name" in event and isinstance(event["name"], str) and event["name"] != ""
    assert (
        "catalog" in event
        and isinstance(event["catalog"], str)
        and event["catalog"] != ""
    )


@pytest.mark.remote
def test_fetch_allowed_params():
    params = apiv2.fetch_allowed_params()
    assert len(params) > 0


def test_retry_on_429(requests_mock):
    """
    Check that we retry for a few `429 Too Many Requests` responses.
    """
    # We reply 429 for several times and then 200. This should work.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_retry_on_429"
    responses = [response_429] * (apiv2.MAX_RETRIES - 1) + [response_200]
    requests_mock.get(mock_url, responses)
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        response = apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES
    assert str(response) == responses[-1]["text"]


def test_fail_too_many_429(requests_mock):
    """
    Check that we eventually fail for `429 Too Many Requests` responses.
    """
    # We reply 429 forever. This should raise a RetryError exception.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_fail_too_many_429"
    requests_mock.get(mock_url, **response_429)
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        with pytest.raises(tenacity.RetryError):
            apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES


def test_retry_on_transient_error(requests_mock):
    """
    Check that we retry for a few transient errors
    """
    # We reply 500 for several times and then 200. This should work.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_retry_on_transient_error"
    responses = [response_500] * (apiv2.MAX_RETRIES - 1) + [response_200]
    requests_mock.get(mock_url, responses)
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        response = apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES
    assert str(response) == responses[-1]["text"]


def test_fail_too_many_transient_error(requests_mock):
    """
    Check that we eventually fail for transient errors.
    """
    # We reply 500 forever. This should raise a RetryError exception.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_fail_too_many_transient_error"
    requests_mock.get(mock_url, **response_500)
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        with pytest.raises(tenacity.RetryError):
            apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES


def test_retry_on_few_timeouts(requests_mock):
    """
    Check that we retry for a few timeouts.
    """
    # We raise a timeout for several times and then 200. This should work.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_retry_on_few_timeouts"
    responses = [exception_timeout] * (apiv2.MAX_RETRIES - 1) + [response_200]
    requests_mock.get(mock_url, responses)
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        response = apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES
    assert str(response) == response_200["text"]


def test_fail_too_many_timeouts(requests_mock):
    """
    Check that we eventually fail for too many timeouts.
    """
    # We raise ReadTimeout forever. This should raise a RetryError exception.
    # We want to test retry so set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_fail_too_many_timeouts"
    exceptions = requests.ReadTimeout
    requests_mock.get(
        mock_url,
        exc=exceptions,
    )
    with patch.object(apiv2, "RETRY_DEFAULT_WAIT_TIME", 0):
        with pytest.raises(tenacity.RetryError):
            apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == apiv2.MAX_RETRIES


def test_fail_on_permanent_http_error(requests_mock):
    """
    Check that we fail immediately on permanent errors.
    """
    # We reply 404 forever. This should raise an HTTPError exception on first attempt.
    # We want to test immediate failure so we don't set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_fail_on_permanent_http_error"
    requests_mock.get(mock_url, **response_404)
    with pytest.raises(requests.HTTPError, match="404 Client Error"):
        apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == 1


def test_fail_on_permanent_request_error(requests_mock):
    """
    Check that we fail immediately on permanent requests errors.
    """
    # We raise SSLError. This should raise the same exception on first attempt.
    # We want to test immediate failure so we don't set RETRY_DEFAULT_WAIT_TIME to 0.
    mock_url = "http://test_fail_on_permanent_request_error"
    requests_mock.get(mock_url, exc=requests.exceptions.SSLError)
    with pytest.raises(requests.exceptions.SSLError):
        apiv2.fetch_json(mock_url)
    assert apiv2._fetch_json.statistics["attempt_number"] == 1
