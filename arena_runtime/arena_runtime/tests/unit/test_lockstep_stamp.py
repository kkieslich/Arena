"""The lockstep gate reads each channel's stamp off the serialized message."""

from __future__ import annotations

import pytest
from arena_humansim_msgs.msg import AgentFrame
from arena_rclpy_mixins.Time import Time
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_runtime_msgs.msg import LockstepHeartbeat
from rclpy.serialization import serialize_message
from std_msgs.msg import String

from arena_runtime.lockstep import stamp_reader


def _stamped(msg_type: type, sec: int, nanosec: int):
    msg = msg_type()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    return msg


@pytest.mark.parametrize("msg_type", [Pedestrians, AgentFrame, LockstepHeartbeat])
def test_the_stamp_comes_off_the_cdr_without_decoding(msg_type: type) -> None:
    msg = _stamped(msg_type, 1234, 50_000_000)
    if msg_type is Pedestrians:
        msg.pedestrians = [Pedestrian(name=f"ped_{i}") for i in range(3)]
    assert stamp_reader(msg_type)(serialize_message(msg)) == Time.from_msg(msg.header.stamp).to_seconds()


def test_a_big_endian_body_reads_the_same_stamp() -> None:
    data = bytearray(serialize_message(_stamped(LockstepHeartbeat, 7, 250_000_000)))
    data[1] = 0
    data[4:8] = (7).to_bytes(4, "big")
    data[8:12] = (250_000_000).to_bytes(4, "big")
    assert stamp_reader(LockstepHeartbeat)(bytes(data)) == 7.25


def test_an_unknown_representation_falls_back_to_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import arena_runtime.lockstep as lockstep

    monkeypatch.setattr(lockstep, "_PLAIN_CDR", {})
    assert stamp_reader(AgentFrame)(serialize_message(_stamped(AgentFrame, 3, 0))) == 3.0


def test_a_type_without_a_leading_header_is_decoded() -> None:
    with pytest.raises(AttributeError):
        stamp_reader(String)(serialize_message(String(data="x")))
