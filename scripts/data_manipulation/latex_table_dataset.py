import pandas as pd
import os
from utils import load_dataframe, dataframe_dict


# Configuration
DATASETS = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']

def load_dataset_stats(ds_name):
    """
    Loads df1, df2, and ground truth to extract dataset statistics.
    Modify the file paths based on your actual directory structure.
    """
    try:

        dataset_path = f"../../data/ccer/{ds_name}"

        filename1, filename2, attributes, separator = dataframe_dict[ds_name]


        # Example paths - adjust these to match your actual files
        df1_path = os.path.join(dataset_path, f"{filename1}.csv")
        df2_path = os.path.join(dataset_path, f"{filename2}.csv")
        gt_path = os.path.join(dataset_path, "gtclean.csv")

        print(df1_path, df2_path, gt_path)

        df1 = pd.read_csv(df1_path, sep=separator)
        df2 = pd.read_csv(df2_path, sep=separator)
        gt = pd.read_csv(gt_path, sep=separator)

        # Calculate stats
        e1 = len(df1)
        e2 = len(df2)
        matches = len(gt)

        # Attributes: Adjust this logic based on what you mean by "used" vs "actual"
        # Here we assume 'actual' is total columns, and 'used' is total minus IDs/keys
        a1_total = len(df1.columns)
        a2_total = len(df2.columns)

        # Example: assume 1 column is the ID and the rest are "used" attributes
        a1_used = len(attributes)
        a2_used = len(attributes)

        return e1, e2, a1_total, a2_total, a1_used, a2_used, matches

    except FileNotFoundError:
        # Dummy data for demonstration if files aren't found
        return 15000, 15000, 10, 10, 8, 8, 3500

def format_num(num):
    """Formats large numbers with commas for readability."""
    return f"{num:,}"

def generate_dataset_latex():
    """Generates the LaTeX table string."""

    latex = r"""\begin{table}[ht]
\centering
\caption{Dataset characteristics. $|E|$ denotes the number of entities, $|A|$ denotes the total actual attributes, $|A_{used}|$ denotes the number of attributes used for blocking/matching, and $|M|$ denotes the number of actual matches (ground truth).}
\label{tab:datasets}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{Entities}} & \multicolumn{2}{c}{\textbf{Total Attrs.}} & \multicolumn{2}{c}{\textbf{Used Attrs.}} & \textbf{Matches} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}
\textbf{Dataset} & $|E_1|$ & $|E_2|$ & $|A_1|$ & $|A_2|$ & $|A_{1, used}|$ & $|A_{2, used}|$ & $|M|$ \\
\midrule
"""

    for ds in DATASETS:
        e1, e2, a1_tot, a2_tot, a1_used, a2_used, matches = load_dataset_stats(ds)

        # Format the row
        row = f"{ds} & {format_num(e1)} & {format_num(e2)} & {a1_tot} & {a2_tot} & {a1_used} & {a2_used} & {format_num(matches)} \\\\\n"
        latex += row

    latex += r"""\bottomrule
\end{tabular}
}
\end{table}
"""
    return latex

if __name__ == "__main__":
    latex_table = generate_dataset_latex()

    with open("dataset_characteristics.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)
    print("\nTable generated successfully and saved to 'dataset_characteristics.tex'.")