"""Fix Clustering PANDAS"""
import os
import ast
import pandas as pd

if __name__ == '__main__':

    RESULTS_DIR = 'results/metablocking/D8/clustering'
    subfolders = [f.path for f in os.scandir(RESULTS_DIR) if f.is_dir()]
    print(subfolders)
    for subfolder in subfolders:
        csv_files = [f.path for f in os.scandir(subfolder)]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['_config'] = df['config'].apply(ast.literal_eval)
            df['num_hashes'] = df['_config'].apply(lambda  x: x.get('num_hashes'))
            df.drop(columns=['_config'], inplace=True)
            df = df[df['num_hashes'] != 25]
            df.drop(columns=['num_hashes'], inplace=True)
            df.to_csv(csv_file, index=False, mode='w+')

    print("Done")