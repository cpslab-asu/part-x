from dataclasses import dataclass
from typing import Any

@dataclass
class Result:
    fv_stats_wo_gp: Any
    fv_stats_with_gp: Any
    falsification_rate: Any
    falsified_true: Any
    first_falsification_mean_mean: Any
    first_falsification_mean_median: Any
    first_falsification_mean_min: Any
    first_falsification_mean_max: Any
    best_robustness: Any
    best_falsification_points: Any
    falsification_points: Any
    non_falsification_points: Any
    
    
