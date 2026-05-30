# Copyright (C) Cardiff University (2018-2021)
# SPDX-License-Identifier: MIT

"""Tests for :mod:`gwosc.datasets`"""

from unittest import mock

import pytest

from .. import datasets

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.mark.remote
def test_find_datasets():
    sets = datasets.find_datasets()
    for dset in ("S6", "O1", "GW150914-v1", "GW170817-v3"):
        assert dset in sets
    assert "tenyear" not in sets
    assert "history" not in sets


@pytest.mark.remote
def test_find_datasets_detector():
    v1sets = datasets.find_datasets("V1")
    assert "GW170817-v3" in v1sets
    assert "GW150914-v1" not in v1sets

    assert datasets.find_datasets("X1", type="run") == []


@pytest.mark.remote
def test_find_datasets_type():
    runsets = datasets.find_datasets(type="run")
    assert "O1" in runsets
    assert datasets.find_datasets(type="badtype") == []


@pytest.mark.remote
def test_find_datasets_segment():
    sets = datasets.find_datasets(segment=(1126051217, 1137254417))
    assert "GW150914-v1" in sets
    assert "GW170817" not in sets


@pytest.mark.remote
def test_find_datasets_match():
    assert "O1" not in datasets.find_datasets(match="GW")


@pytest.mark.remote
def test_find_datasets_event_version_detector():
    # this raises a ValueError with gwosc-0.5.0
    sets = datasets.find_datasets(type="event", version=1, detector="L1")
    assert "GW150914-v1" in sets
    assert "GW150914-v3" not in sets  # v3


@mock.patch("gwosc.datasets._run_datasets", return_value=[])
def test_find_datasets_warning(_):
    with pytest.warns(UserWarning):
        datasets.find_datasets(type="run", version=1)


@pytest.mark.remote
def test_event_gps():
    assert datasets.event_gps("GW170817") == 1187008882.4


@mock.patch(
    "gwosc.api.v2.fetch_event_version",
    return_value={
        "name": "GW150914",
        "gps": 12345.6,
        "something_else": 1,
    },
)
def test_event_gps_local(fetch):
    assert datasets.event_gps("GW150914") == 12345.6


@pytest.mark.remote
def test_event_segment():
    assert datasets.event_segment("GW170817") == (1187006835, 1187010931)


@mock.patch(
    "gwosc.api.v2.fetch_event_strain_data",
    mock.MagicMock(
        return_value=[
            {"gps_start": 0, "detector": "X1", "duration": 32},
            {"gps_start": 10, "detector": "Y1", "duration": 32},
        ]
    ),
)
def test_event_segment_local_without_det():
    assert datasets.event_segment("GW170817") == (0, 42)


@pytest.mark.remote
def test_event_at_gps():
    assert datasets.event_at_gps(1187008882) == "GW170817"
    with pytest.raises(ValueError) as exc:
        datasets.event_at_gps(1187008882, tol=0.1)
    assert str(exc.value) == "no event found within 0.1 seconds of 1187008882"


@mock.patch(
    "gwosc.api.v2.fetch_event_versions",
    mock.MagicMock(
        return_value=[
            {
                "name": "GW150914",
                "gps": 12345,
            }
        ]
    ),
)
def test_event_at_gps_local_found():
    assert datasets.event_at_gps(12345) == "GW150914"


@mock.patch(
    "gwosc.api.v2.fetch_event_versions",
    mock.MagicMock(return_value=[]),
)
def test_event_at_gps_local_not_found():
    with pytest.raises(ValueError):
        datasets.event_at_gps(12349)


@pytest.mark.remote
def test_event_detectors():
    assert datasets.event_detectors("GW150914") == {"H1", "L1"}
    assert datasets.event_detectors("GW170814") == {"H1", "L1", "V1"}


@mock.patch(
    "gwosc.api.v2.fetch_event_version",
    mock.MagicMock(
        return_value={
            "name": "GW150914",
            "gps": 12345.6,
            "detectors": ["A1", "B1"],
        }
    ),
)
def test_event_detectors_local():
    assert datasets.event_detectors("test") == {"A1", "B1"}


@pytest.mark.remote
def test_run_segment():
    assert datasets.run_segment("O1") == (1126051217, 1137254417)
    with pytest.raises(ValueError):
        datasets.run_segment("S7")


@mock.patch(
    "gwosc.api.v2.fetch_run",
    mock.MagicMock(return_value={"gps_start": 0, "gps_end": 1}),
)
def test_run_segment_local():
    assert datasets.run_segment("S1") == (0, 1)


@pytest.mark.remote
def test_run_at_gps():
    assert datasets.run_at_gps(1135136350) in {"O1", "O1_16KHZ"}
    with pytest.raises(ValueError) as exc:
        datasets.run_at_gps(0)
    assert str(exc.value) == "no run dataset found containing GPS 0"


@mock.patch(
    "gwosc.api.v2.fetch_runs",
    mock.MagicMock(
        return_value=[
            {"name": "S1", "gps_start": 0, "gps_end": 1},
            {"name": "S2", "gps_start": 3, "gps_end": 4},
        ]
    ),
)
def test_run_at_gps_local():
    assert datasets.run_at_gps(0) == "S1"
    with pytest.raises(ValueError):
        datasets.run_at_gps(10)


@pytest.mark.remote
def test_dataset_type():
    assert datasets.dataset_type("O1") == "run"
    assert datasets.dataset_type("GW150914-v1") == "event"
    assert datasets.dataset_type("GWTC-1-confident") == "catalog"
    with pytest.raises(ValueError):
        datasets.dataset_type("invalid")


@mock.patch(
    "gwosc.datasets.find_datasets",
    mock.MagicMock(side_effect=[["testrun"], [], ["testevent"], [], [], []]),
)
def test_dataset_type_local():
    assert datasets.dataset_type("testevent") == "event"
    with pytest.raises(ValueError):
        datasets.dataset_type("invalid")


@pytest.mark.remote
def test_query_events():
    events = datasets.query_events(select=["10 <= luminosity-distance <= 200"])
    assert "GW190425-v1" in events
    assert "GW190425-v2" in events
    assert "GW190425_081805-v3" in events


@mock.patch(
    "gwosc.api.v2.fetch_event_versions",
    mock.MagicMock(
        return_value=[
            {
                "name": "mock-event",
                "gps": 1240215503.0,
                "version": 1,
                "shortName": "mock-event-1",
            },
            {
                "name": "mock-event",
                "gps": 1240215503.0,
                "version": 2,
                "shortName": "mock-event-2",
            },
            {
                "name": "mock-event",
                "gps": 1240215503.0,
                "version": 3,
                "shortName": "mock-event-3",
            },
        ]
    ),
)
def test_query_events_local():
    events = datasets.query_events(
        select=["mass-1-source >= 1.4", "10 <= luminosity-distance <= 200"]
    )
    assert "mock-event-1" in events
    assert "mock-event-2" in events
    assert "mock-event-3" in events
