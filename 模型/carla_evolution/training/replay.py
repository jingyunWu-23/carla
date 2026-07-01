"""Task-level replay support for penetration-stage forgetting evaluation."""


class StageReplayBuffer:
    """Stores historical penetration-stage configurations and metrics."""

    def __init__(self):
        self.stages = []

    def add(self, stage_config, metrics):
        self.stages.append({"stage_config": stage_config, "metrics": metrics})

