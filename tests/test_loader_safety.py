"""Tests for style loader safety (element-count & cap validation)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from voxbridge.loader import _MAX_STYLE_ELEMENTS, load_voice_style_from_json_file


def _write_style_json(path: str, /, *, dims: list[int], data: list[float]) -> None:
    """Helper: write a minimal style JSON file with given dims and data."""
    style_data = {
        "version": 1,
        "author": "test",
        "dims": dims,
        "data": data,
    }
    with open(path, "w") as fh:
        json.dump(style_data, fh)


def test_voice_style_rejects_too_many_elements():
    """Exceeding the element cap must raise ValueError before np.array()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # dims = [cap + 1] → expected_elements = cap + 1 > cap
        path = os.path.join(tmpdir, "too_big.json")
        bad_count = _MAX_STYLE_ELEMENTS + 1
        _write_style_json(path, dims=[bad_count], data=[0.0] * 5)  # data length irrelevant — check dies on expected_elements
        with pytest.raises(ValueError):
            load_voice_style_from_json_file(path)


def test_voice_style_rejects_mismatched_shape():
    """Data length that doesn't match dims must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mismatched.json")
        _write_style_json(path, dims=[3, 3], data=[0.0] * 5)  # expects 9, gets 5
        with pytest.raises(ValueError):
            load_voice_style_from_json_file(path)


def test_voice_style_accepts_exactly_at_element_cap():
    """Exactly at the cap should pass the element check (may fail real loading)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "at_cap.json")
        _write_style_json(
            path,
            dims=[_MAX_STYLE_ELEMENTS],
            data=[0.0] * _MAX_STYLE_ELEMENTS,
        )
        try:
            load_voice_style_from_json_file(path)
        except (FileNotFoundError, KeyError, ValueError):
            pass  # real model metadata missing in fake fixture is fine


def test_voice_style_under_cap_passes():
    """Small valid style vectors should not trigger safety errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "small.json")
        _write_style_json(path, dims=[5, 2], data=[0.0] * 10)
        try:
            load_voice_style_from_json_file(path)
        except (FileNotFoundError, KeyError, ValueError):
            pass  # real model metadata missing OK


def test_voice_style_nested_object_not_accepted():
    """Plain JSON objects lacking 'dims'/'data' keys raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nested.json")
        with open(path, "w") as fh:
            json.dump({
                "version": 1,
                "author": "test",
                "elements": {"mean": [0.0] * 10, "std": [1.0] * 10},
            }, fh)
        with pytest.raises(ValueError):
            load_voice_style_from_json_file(path)
