from dataclasses import dataclass
from typing import Any

@dataclass
class Result:
    con_int_fv_wo_gp_classified_quan_confidence_0_95: Any
    con_int_fv_wo_gp_classified_unclassified_quan_confidence_0_95: Any
    mean_fv_wo_gp_classified: Any
    mean_fv_wo_gp_classified_unclassified: Any
    std_dev_fv_wo_gp_classified: Any
    std_dev_fv_wo_gp_classified_unclassified: Any
    falsified_true: Any
    numpoints_fin_first_f_mean: Any
    numpoints_fin_first_f_median: Any
    numpoints_fin_first_f_min: Any
    numpoints_fin_first_f_max: Any
    falsification_rate: Any
    best_robustness: Any
    mean_fv_with_gp_quan0_5: Any
    std_dev_fv_with_gp_quan0_5: Any
    con_int_fv_with_gp_quan_0_5_confidence_0_95: Any
    mean_fv_with_gp_quan0_95: Any
    std_dev_fv_with_gp_quan0_95: Any
    con_int_fv_with_gp_quan_0_95_confidence_0_95: Any
    mean_fv_with_gp_quan0_99: Any
    std_dev_fv_with_gp_quan0_99: Any
    con_int_fv_with_gp_quan_0_99_confidence_0_95: Any
    falsification_corr_point: Any
    unfalsification_corr_point: Any
    