"""Task variance — moved to legacy.  Re-exported for backward compatibility."""

from icl.linear.legacy.task_variance import (  # noqa: F401
    TaskVarianceResults,
    compute_task_variance,
    compute_task_variance_multi_layer,
    results_to_dict,
    results_to_table,
    save_results_json,
    save_results_multi_layer_json,
    extract_plotting_data,
    extract_plotting_data_multi_layer,
)
