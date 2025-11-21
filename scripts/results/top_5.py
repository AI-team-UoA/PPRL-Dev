"""TOP 5"""

import pandas as pd


top_results_list = []

for filename in ['CardinalityEdgePruning.csv',
            'faiss.csv', 'WeightedEdgePruning.csv',
             'WeightedNodePruning.csv']:
    df = pd.read_csv(f'results/{filename}')
    # Sort the DataFrame

    df  = df[df['candidate_pairs'] < 11_000]

    df_sorted = df.sort_values(
        by=['Recall %', 'candidate_pairs'],
        ascending=[False, True]
    )

    # 6. Get the top 5 rows
    top_5_df = df_sorted.head(5)


    # 8. Append the top 5 results to the list
    top_results_list.append(top_5_df)
    # 9. Concatenate all the individual DataFrames into one final DataFrame


if top_results_list:
    final_df = pd.concat(top_results_list, ignore_index=True)

    print(final_df.iloc[0])

    final_df.to_csv('results/top_5.csv', mode='w+', index=False)

