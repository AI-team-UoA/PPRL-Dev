import pandas as pd

datasets = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']
filtering = 'results/random_configurations/filtering'
metablocking = 'results/random_configurations/metablocking'
metablocking_methods = ['BLAST.csv', 'CardinalityEdgePruning.csv', 'CardinalityNodePruning.csv', 'WeightedEdgePruning.csv', 'WeightedNodePruning.csv']

weighting_schemes = ['CN-CBS', 'CBS', 'SN-CBS',
    'SNC', 'CNC', 'SND', 'CND', 'CNJ', 'SNJ',
    'COSINE', 'DICE', 'ECBS', 'JS', 'EJS', 'X2']


for dataset in datasets:
    if dataset == 'D10':
        continue
    filtering_df = pd.read_csv(f'{filtering}/{dataset}/block_filtering.csv')
    filtering_runtime = filtering_df['total_runtime'].values
    for metablocking_filename in metablocking_methods:
        metablocking_df = pd.read_csv(f'{metablocking}/{dataset}/{metablocking_filename}')
        metablocking_runtime = metablocking_df['runtime'].values
        metablocking_total_runtime = []


        for filtering_idx, f_runtime in enumerate(filtering_runtime):
            for i, _ in enumerate(weighting_schemes):
                metablocking_total_runtime.append(f_runtime + metablocking_runtime[filtering_idx * len(weighting_schemes) + i])
        metablocking_df['total_runtime'] = metablocking_total_runtime
        cols = list(metablocking_df.columns)

        cols.remove('total_runtime')
        insert_loc = cols.index('runtime') + 1
        cols.insert(insert_loc, 'total_runtime')
        metablocking_df = metablocking_df[cols]
        metablocking_df.to_csv(f'{metablocking}/{dataset}/{metablocking_filename}', index=False, float_format='%.2f')



