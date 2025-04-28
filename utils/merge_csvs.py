import os
import glob
import pandas as pd

from utils.load_dotenv import load_dotenv

load_dotenv()

def merge_csvs(merge_dir: str, output_dir: str, output_file = "merged.csv") -> None:

    if not os.path.exists(merge_dir):
        raise FileNotFoundError(f"Directory {merge_dir} not found")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_file)

    csv_files = glob.glob(os.path.join(merge_csvs, "*.csv"))

    dfs = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])  # Adjust column name if different
    merged_df = merged_df.sort_values('datetime')
    merged_df.to_csv(output_path, index=False)
