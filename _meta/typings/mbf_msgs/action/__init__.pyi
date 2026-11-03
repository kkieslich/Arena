from typing import ClassVar

from geometry_msgs.msg import PoseStamped

class MoveBase:
    class Goal:
        target_pose: PoseStamped
        controller: str
        planner: str
        recovery_behaviors: list[str]

    class Result:
        SUCCESS: ClassVar[int]
        CANCELED: ClassVar[int]
        outcome: int
        message: str
        final_pose: PoseStamped

    class Feedback:
        current_pose: PoseStamped
        dist_to_goal: float

    class Impl:
        class FeedbackMessage:
            feedback: MoveBase.Feedback
