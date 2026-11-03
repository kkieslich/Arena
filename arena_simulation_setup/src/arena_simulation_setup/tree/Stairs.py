"""Stairs kinds: named presets of how a stair or escalator is built and dressed, resolved like wall kinds."""

from __future__ import annotations

from pathlib import Path

import yaml

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree import DomainAssetIdentifier, DynamicPaths, NetResolver, SimplePathResolver

STAIRS_KIND_KEYS = frozenset({'style', 'speed', 'tread', 'landing_length', 'plank', 'headroom', 'material', 'surfaces', 'sides', 'rails', 'models'})


class StairsIdentifier(DomainAssetIdentifier[dict]):
    _asset_type = 'Stairs'
    NESTS = True

    def load(self, path: Path, /, **kwargs: object) -> dict:
        del kwargs  # unused
        with open(path / f'{path.name}.yaml') as f:
            preset = yaml.safe_load(f)
        if not isinstance(preset, dict):
            raise ValueError(f'stair kind {self.name!r}: expected a mapping of stair keys, got {preset!r}')
        return preset


StairsIdentifier.use(*DynamicPaths.as_resolvers(StairsIdentifier))
StairsIdentifier.use(SimplePathResolver(StairsIdentifier, ASS_DIR / 'assets'))
StairsIdentifier.use(*NetResolver.all(StairsIdentifier))
