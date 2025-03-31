import os
import sys
from scipy.io import arff
import pandas as pd

def arff_to_csv(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".arff"):
                filepath = os.path.join(root, file)
                csv_file_path = f"{os.path.splitext(filepath)[0]}.csv"
                if not file_exists(csv_file_path): 
                    data, _ = arff.loadarff(filepath)
                    df = pd.DataFrame(data)
                    df.to_csv(csv_file_path, index=False)
                
def file_exists(name):
    return os.path.exists(name) and os.path.isfile(name)

if len(sys.argv) !=2:
    print("Usage: python arff_to_csv.py <arfffolder>")
    
else:
    folder_path = str(sys.argv[1])
    arff_to_csv(folder_path)