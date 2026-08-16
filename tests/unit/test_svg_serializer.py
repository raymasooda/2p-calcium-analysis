"""Tests for the SVG artifact serializer."""

from __future__ import annotations

from pathlib import Path

from calcium2p.artifacts.store import SvgSerializer, infer_serializer

SVG_DOC = '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'


class TestSvgSerializer:
    def test_round_trip(self, tmp_path: Path) -> None:
        serializer = SvgSerializer()
        target = tmp_path / "figure.svg"
        serializer.save(SVG_DOC, target)
        assert serializer.load(target) == SVG_DOC

    def test_handles_svg_strings_only(self) -> None:
        assert SvgSerializer.handles(SVG_DOC)
        assert SvgSerializer.handles("  <svg xmlns='x'></svg>")
        assert not SvgSerializer.handles("just a string")
        assert not SvgSerializer.handles({"not": "a string"})

    def test_infer_prefers_svg_over_json_for_markup(self) -> None:
        # any string survives a JSON round-trip, so ordering matters
        assert infer_serializer(SVG_DOC) is SvgSerializer

    def test_estimate_size_is_exact(self) -> None:
        assert SvgSerializer.estimate_size(SVG_DOC) == len(SVG_DOC.encode("utf-8"))
