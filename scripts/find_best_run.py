import pandas as pd


df = pd.read_csv('find_best_run.csv')
df = df.sort_values(by=['matcher_F1 %','Recall %'], ascending=[False, False])
row = df.iloc[0]
print(row)
print(row['cofig'])
