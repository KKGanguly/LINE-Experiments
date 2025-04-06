DATASETS_DIR = data/moot_test/optimize/
COMMAND_FILE = commands.sh
BASE_CMD = python3.13 experiments/experiement_runner_parallel.py --name DEHB --repeats 20 --budget 6 12 18 24 50 100 200 --runs_output_folder results_DEHB_parallel --logging_folder logging_DEHB_parallel --output_directory dehb_tmp_runs

generate-commands:
	@echo "#!/bin/bash" > $(COMMAND_FILE)
	@find $(DATASETS_DIR) -type f -name "*.csv" | while read dataset; do \
		echo "$(BASE_CMD) --datasets $$dataset &" >> $(COMMAND_FILE); \
	done
	@echo "wait" >> $(COMMAND_FILE)
	@chmod +x $(COMMAND_FILE)

run-commands:
	@nohup ./$(COMMAND_FILE) > run.log 2>&1 &
	@echo "Commands are running in the background. Output is in run.log"