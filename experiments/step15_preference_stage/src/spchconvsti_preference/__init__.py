"""论文第二阶段：多维评分投影头的偏好优化。"""

from .stage_two import PreferenceNoisyPair, StageTwoPreferenceObjective, freeze_modules
from .trainer import InBatchPreferenceTrainer, TwoTowerProjectionHead

__all__ = ["InBatchPreferenceTrainer", "TwoTowerProjectionHead", "PreferenceNoisyPair", "StageTwoPreferenceObjective", "freeze_modules"]
