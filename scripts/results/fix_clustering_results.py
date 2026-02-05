"""Fix Clustering PANDAS"""

import ast
import pandas as pd

if __name__ == '__main__':


    TOLERANCE = 2.0
    for i in [2,3,4,8]:
        DIR = f'D{i}'


        CLUSTERING_TYPE = 'KiralyMSMApproximateClustering'
        RESULTS_DIR = f'results/clustering/{DIR}'
        ev_columns = ['Precision %', 'Recall %', 'F1 %']
        for metric in ['cosine', 'dice', 'scm', 'jaccard']:
            if metric == 'scm' and DIR == 'D8':
                continue
            # METRIC = 'cosine'
            filename = f'{RESULTS_DIR}/{CLUSTERING_TYPE}_{metric}.csv'
            df = pd.read_csv(filename)

            df['_config'] = df['config'].apply(ast.literal_eval)
            df['num_hashes'] = df['_config'].apply(lambda  x: x.get('num_hashes'))
            df.drop(columns=['_config'], inplace=True)
            df = df[df['matching_thr'] == df['clustering_threshold']]
            df.drop(columns=['matching_thr'], inplace=True)
            df.rename(columns={'clustering_threshold' : 'threshold'}, inplace=True)


            df.drop_duplicates(subset=['threshold', 'num_hashes'],
                            inplace=True,
                            keep='first')
            df.drop(columns=['num_hashes'], inplace=True)
            df.to_csv(filename, index=False, mode='w+')
            print("Inconsitency not found")



    # df = df.sort_values(by=['num_hashes', 'threshold'], ascending=True)
    # print(df[['threshold', 'F1 %', 'num_hashes']].head(50))
