"""The gait library is read once per process, by libyaml when PyYAML has it, and a cold first call from two threads parses it once."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yaml

from arena_runtime.humans import profile


def _builtin_agent_types() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("arena_humansim")) / "config" / "agent_types"
    except Exception:  # noqa: BLE001  (no humansim in this workspace)
        return None


def test_a_cold_first_call_from_two_threads_parses_the_library_once(monkeypatch: pytest.MonkeyPatch) -> None:
    files = sorted(Path(profile.LIBRARY_DIR).glob("*.yaml"))
    assert files
    parsed: list[str] = []
    read = profile.read_yaml

    def slow_read(fh: object) -> object:
        parsed.append(getattr(fh, "name", "?"))
        time.sleep(0.01)  # long enough for the second thread to arrive while the first one reads
        return read(fh)

    monkeypatch.setattr(profile, "read_yaml", slow_read)
    monkeypatch.setattr(profile, "_LIBRARIES", {})
    start = threading.Barrier(2)
    got: list[object] = []

    def call() -> None:
        start.wait()
        got.append(profile.load_library())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(parsed) == len(files)
    assert got[0] is got[1]


def test_libyaml_reads_every_gait_profile_and_builtin_agent_type_as_safe_load_does() -> None:
    paths = sorted(Path(profile.LIBRARY_DIR).glob("*.yaml"))
    builtin = _builtin_agent_types()
    if builtin is not None and builtin.is_dir():
        paths += sorted(builtin.glob("*.yaml"))
    for path in paths:
        with open(path) as fh:
            fast = profile.read_yaml(fh)
        with open(path) as fh:
            assert fast == yaml.safe_load(fh), path


def test_the_loader_is_libyaml_when_pyyaml_has_it() -> None:
    assert profile._YAML_LOADER is (yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.SafeLoader)
