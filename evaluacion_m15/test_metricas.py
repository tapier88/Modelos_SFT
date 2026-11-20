"""Analytical checks for scoring and day-cluster confidence intervals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import evaluar_base_m15 as ev

rows = []
wid = 0
for year in [2020, 2021, 2022]:
    for day in [2, 3]:
        for ny in [False, True]:
            for h in range(1, 5):
                rows.append(dict(window_id=wid, session_date_ny=f'{year}-01-{day:02d}', year_ny=year,
                                 ny_full_horizon=ny, horizon=h, actual_close=101. if ny else 99.,
                                 predicted_close=100.5 if ny else 99.5, persistence_close=100.))
            wid += 1
summary, _, _ = ev.summarize(pd.DataFrame(rows))
for column, expected in [('close_mae', .5), ('persistence_close_mae', 1.), ('mae_skill_pct', 50.),
                         ('direction_accuracy_nonflat', 1.), ('delta_ci95_low', -.5), ('delta_ci95_high', -.5)]:
    assert np.allclose(summary[column], expected), (column, summary[column])
single_day = pd.DataFrame({'session_date_ny': ['2020-01-01'], 'delta_mae': [1.]})
assert ev.cluster_interval(single_day) == [None, None]
checks = ev.self_test()
checks.update(mae_exact=True, skill_exact=True, direction_exact=True, paired_bootstrap_exact=True,
              all_groups_exercised=True, single_day_interval_unavailable=True)
output_dir = Path(__file__).parent/'resultados_20261111'
if output_dir.is_dir():
    (output_dir/'metric_tests.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print('METRIC_TESTS_PASSED', json.dumps(checks))
