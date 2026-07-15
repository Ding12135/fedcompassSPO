"""SU-Compass 诊断工具包。

本包中的模块只负责观测、配对和汇总实验事实，不参与 Q 分配、arrival group
选择或模型聚合。将诊断逻辑与调度策略分开，可以保证后续替换预测方法时，
论文问题证据仍由同一套独立观测口径产生。
"""

from .fedcompass_problem import FedCompassProblemObserver

__all__ = ["FedCompassProblemObserver"]
