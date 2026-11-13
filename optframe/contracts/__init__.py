"""共享合同层：两档必须共用同一套任务、judge、计时与账本。"""

from .evaluator import Baseline, evaluate_candidate, measure_baseline
from .ledger import Ledger
from .schema import Candidate, Result, Verdict
from .source_guard import check_candidate
from .task import Task, TaskError, load_task
from .timing import time_ms

__all__ = [
    "Baseline",
    "Candidate",
    "Ledger",
    "Result",
    "Task",
    "TaskError",
    "Verdict",
    "check_candidate",
    "evaluate_candidate",
    "load_task",
    "measure_baseline",
    "time_ms",
]
