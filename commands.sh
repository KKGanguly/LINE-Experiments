#!/bin/bash
python3.13 experiments/experiement_runner_parallel.py --name DEHB --repeats 20 --budget 6 12 18 24 50 100 200 --runs_output_folder results_DEHB_parallel --logging_folder logging_DEHB_parallel --output_directory dehb_tmp_runs --datasets data/moot_test/optimize/config/SS-F.csv &
python3.13 experiments/experiement_runner_parallel.py --name DEHB --repeats 20 --budget 6 12 18 24 50 100 200 --runs_output_folder results_DEHB_parallel --logging_folder logging_DEHB_parallel --output_directory dehb_tmp_runs --datasets data/moot_test/optimize/config/SS-G.csv &
wait
