import pandas as pd
import os

class Logger:
    def __init__(self, log_file='optimization_log.csv'):
        self.log_file = log_file
        # Ensure the log file exists with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            # Create a new DataFrame and save it as a CSV with headers
            df = pd.DataFrame(columns=['best_config', 'best_value'])
            df.to_csv(self.log_file, index=False)

    def log(self, best_config, best_value):
        # Create a DataFrame from the best configuration and value
        new_log_entry = pd.DataFrame({
            'best_config': [best_config],
            'best_value': [best_value]
        })
