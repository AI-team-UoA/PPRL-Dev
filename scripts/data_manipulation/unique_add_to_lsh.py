import pandas as pd


datasets = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']
old_filtering = 'results/random_configurations/filtering'
new_filtering = 'results/random_configurations/unique_blocking/filtering'

# lsh_blocker = 'results/random_configurations/unique_blocking/lsh_blocker'


filename = 'block_filtering.csv'



for dataset in datasets:
    if dataset == 'D10':
        continue

    old_df = pd.read_csv(f'{old_filtering}/{dataset}/{filename}')
    new_df = pd.read_csv(f'{new_filtering}/{dataset}/{filename}')
    # lsh_df = pd.read_csv(f'{lsh_blocker}/{dataset}/lsh_blocker.csv')
    # lsh_runtime = lsh_df['runtime'].values

    # unique_candidate_pairs = new_df['candidate_pairs'].values




    # old_df['unique_candidate_pairs'] = unique_candidate_pairs


# 
    # old_df['total_runtime'] = new_df['total_runtime'].values
    # old_df['runtime'] = new_df['runtime'].values

    old_df['Unique Precision %'] = new_df['Precision %'].values
    old_df['Unique F1 %'] = new_df['F1 %'].values


    cols = list(old_df.columns)



    # cols.remove('unique_candidate_pairs')
    # cols.remove('total_runtime')

    cols.remove('Unique Precision %')
    cols.remove('Unique F1 %')



    # insert_loc = cols.index('candidate_pairs') + 1
    # cols.insert(insert_loc, 'total_runtime')
    # cols.insert(insert_loc + 1, 'unique_candidate_pairs')


    insert_loc = cols.index('F1 %') + 1
    cols.insert(insert_loc, 'Unique Precision %')
    cols.insert(insert_loc + 1, 'Unique F1 %')

    old_df = old_df[cols]

    old_df.to_csv(f'{old_filtering}/{dataset}/{filename}', index=False, float_format='%.2f')