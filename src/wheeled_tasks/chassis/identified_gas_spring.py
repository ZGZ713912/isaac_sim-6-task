"""Bounded effective spring force identified from LEFT measured motor efforts.

This optional curve does not replace the catalogue/default training model.
Compression is positive inward; returned positive axial force acts outward.
Right-side use requires an explicit symmetry assumption, not right-side data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class IdentifiedGasSpringCurve:
    compression_knots_m: tuple[float, ...]
    force_knots_n: tuple[float, ...]
    side: str = "left"
    symmetry_assumed: bool = False

    def __post_init__(self):
        if (self.side not in ("left", "right") or (self.side == "right" and not self.symmetry_assumed)
                or len(self.compression_knots_m) < 2
                or len(self.compression_knots_m) != len(self.force_knots_n)
                or not all(math.isfinite(x) for x in self.compression_knots_m + self.force_knots_n)
                or self.compression_knots_m[0] < 0
                or any(b <= a for a, b in zip(self.compression_knots_m, self.compression_knots_m[1:]))
                or any(f <= 0 for f in self.force_knots_n)
                or any(b < a for a, b in zip(self.force_knots_n, self.force_knots_n[1:]))):
            raise ValueError("Need positive nondecreasing force; right side requires explicit symmetry assumption")

    def for_side(self, side: str, *, assume_symmetry: bool = False):
        """Reuse axial force on the mirrored leg; its geometry supplies torque signs."""
        if side == self.side:
            return self
        if side != "right" or self.side != "left" or not assume_symmetry:
            raise ValueError("Mirroring LEFT data to right requires assume_symmetry=True")
        return type(self)(self.compression_knots_m, self.force_knots_n, side="right", symmetry_assumed=True)

    @classmethod
    def from_json(cls, path: str | Path, *, side: str | None = None, assume_symmetry: bool = False):
        data = json.loads(Path(path).read_text())
        model = data["models"][data["selected"]] if "models" in data else data
        if model["kind"] != "effective_monotone_table":
            raise ValueError("Expected the bounded tabulated effective-force model")
        curve = cls(tuple(model["compression_knots_m"]), tuple(model["force_knots_n"]), model["side"])
        return curve.for_side(side, assume_symmetry=assume_symmetry) if side is not None else curve

    def _segments(self, compression_m):
        s = torch.as_tensor(compression_m)
        if not s.is_floating_point():
            s = s.float()
        knots = s.new_tensor(self.compression_knots_m)
        values = s.new_tensor(self.force_knots_n)
        if (not bool(torch.isfinite(s).all()) or bool((s < knots[0]).any())
                or bool((s > knots[-1]).any())):
            raise ValueError("Compression outside identified domain; extrapolation is not implicit")
        index = torch.searchsorted(knots, s.contiguous(), right=True).sub(1).clamp(0, len(knots)-2)
        slope = (values[1:]-values[:-1])/(knots[1:]-knots[:-1])
        return s, knots, values, index, slope

    def force(self, compression_m):
        """Effective extension force [N], not a uniquely measured gas pressure."""
        s, knots, values, index, slope = self._segments(compression_m)
        return values[index]+slope[index]*(s-knots[index])

    def potential_energy(self, compression_m):
        """Stored work [J] relative to the LOWER fitted compression, not full extension."""
        s, knots, values, index, slope = self._segments(compression_m)
        area = (values[1:]+values[:-1])*.5*(knots[1:]-knots[:-1])
        prefix = torch.cat((area.new_zeros(1), area.cumsum(0)))
        delta = s-knots[index]
        return prefix[index]+values[index]*delta+.5*slope[index]*delta.square()
