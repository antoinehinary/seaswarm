"""Refusals: why a command was not executed, in words a human (or an LLM) can act on.

Adapted from skypilot-agent's Decision pattern, with an enum code for programs and a one-sentence
reason for people.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RefusalCode(StrEnum):
    """Which safety rule refused a command."""

    GEOFENCE = "GEOFENCE"
    SEPARATION = "SEPARATION"
    ENERGY_RESERVE = "ENERGY_RESERVE"
    TILE_OCCUPIED = "TILE_OCCUPIED"
    VEHICLE_UNAVAILABLE = "VEHICLE_UNAVAILABLE"


@dataclass(frozen=True)
class Refusal:
    """A refused command: rule code, one-sentence reason, vehicle concerned."""

    code: RefusalCode
    reason: str
    vehicle_id: str | None = None
