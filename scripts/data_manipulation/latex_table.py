import pandas as pd
import os

# Configuration
DATA_DIR = 'seed_runs'
DATASETS = ['D2', 'D3', 'D4', 'D8', 'D9', 'D10']
METHODS = ['LSH', 'LSH+BP', 'LSH+BF', 'LSH+BP+BF']
TARGETS = [0.85, 0.90, 0.95]

def load_data():
    """
    Loads CSV files and computes the mean and std across the 10 seeds based on row indices.
    - Rows 0-9: tau = 0.85
    - Rows 10-19: tau = 0.90
    - Rows 20-29: tau = 0.95
    """
    results = {}
    row_ranges = {0.85: (0, 10), 0.90: (10, 20), 0.95: (20, 30)}

    for ds in DATASETS:
        results[ds] = {}
        for tau, (start_idx, end_idx) in row_ranges.items():
            results[ds][tau] = {}
            for method in METHODS:
                file_path = os.path.join(DATA_DIR, f"{ds}_{method}.csv")

                try:
                    df = pd.read_csv(file_path)
                    df_tau = df.iloc[start_idx:end_idx]

                    if df_tau.empty or len(df_tau) < 2:
                        print(f"Warning: Not enough rows for target {tau} in {file_path}")
                        results[ds][tau][method] = {
                            'recall': 0.0, 'precision': 0.0, 'candidates': 0.0, 'time': 0.0,
                            'recall_std': 0.0, 'precision_std': 0.0, 'candidates_std': 0.0, 'time_std': 0.0
                        }
                        continue

                    results[ds][tau][method] = {
                        'recall': df_tau['Recall %'].mean(),
                        'precision': df_tau['Unique Precision %'].mean(),
                        'candidates': df_tau['Unique Candidate pairs'].mean(),
                        'time': df_tau['Runtime'].mean(),
                        'recall_std': df_tau['Recall %'].std() if pd.notna(df_tau['Recall %'].std()) else 0.0,
                        'precision_std': df_tau['Unique Precision %'].std() if pd.notna(df_tau['Unique Precision %'].std()) else 0.0,
                        'candidates_std': df_tau['Unique Candidate pairs'].std() if pd.notna(df_tau['Unique Candidate pairs'].std()) else 0.0,
                        'time_std': df_tau['Runtime'].std() if pd.notna(df_tau['Runtime'].std()) else 0.0
                    }

                except FileNotFoundError:
                    print(f"Warning: {file_path} not found. Inserting zeros.")
                    results[ds][tau][method] = {
                        'recall': 0.0, 'precision': 0.0, 'candidates': 0.0, 'time': 0.0,
                        'recall_std': 0.0, 'precision_std': 0.0, 'candidates_std': 0.0, 'time_std': 0.0
                    }

    return results

def format_number(val, is_cand=False):
    """Formats numbers. Uses scientific notation or commas for large numbers."""
    if is_cand:
        if val >= 10000:
            s = f"{val:.2e}"
            mantissa, exp = s.split('e')
            exp = int(exp) # removes + and leading zeros
            return f"{mantissa}\\times 10^{{{exp}}}"
        else:
            return f"{val:,.0f}".replace(",", "{,}")
    else:
        return f"{val:.2f}"

def format_mean(val, is_dagger=False, is_cand=False):
    """Formats the mean value string with Math blocks and potential daggers."""
    m_str = format_number(val, is_cand)
    dagger = r"\dagger" if is_dagger else ""
    return f"${m_str}{dagger}$"

def format_std(val, is_cand=False):
    """Formats the standard deviation string using pm."""
    s_str = format_number(val, is_cand)
    return f"$\\pm {s_str}$"

def generate_latex(results):
    """Generates the updated LaTeX table code."""

    latex = r"""\begin{table*}[t]
\centering
\caption{Results for target recall levels (a) $\tau_1 = 0.85$, (b) $\tau_2 = 0.90$ and (c) $\tau_3 = 0.95$. For each method and dataset, the reported configuration maximizes unique precision subject to recall being higher than or equal to the corresponding target. $\dagger$ denotes that no sampled configuration attained the target. In this case, the configuration with the maximum achieved recall is reported instead. Standard deviations are reported below the means.}
\label{tab:step1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrrrr}
\hline
& \multicolumn{4}{c}{$\tau = 0.85$}& \multicolumn{4}{c}{$\tau = 0.90$}& \multicolumn{4}{c}{$\tau = 0.95$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
Dat. & Recall & Prec. & $\lVert C\rVert$ & Time & Recall & Prec. & $\lVert C\rVert$ & Time & Recall & Prec. & $\lVert C\rVert$ & Time \\
\hline"""

    for method in METHODS:
        # Add the Method Header Block
        latex += f"\n\\multicolumn{{13}}{{l}}{{\\textsc{{{method}}}}} \\\\\n\\midrule\n"

        for ds in DATASETS:
            mean_row_str = f"{ds}"
            std_row_str = "" # Empty first column for standard deviation row

            for tau in TARGETS:
                metrics = results[ds][tau][method]

                rec, rec_std = metrics['recall'], metrics['recall_std']
                prec, prec_std = metrics['precision'], metrics['precision_std']
                cand, cand_std = metrics['candidates'], metrics['candidates_std']
                time, time_std = metrics['time'], metrics['time_std']

                # Check if target was missed based on mean
                is_dagger = rec < (tau * 100)

                # Format means
                f_rec = format_mean(rec, is_dagger=is_dagger)
                f_prec = format_mean(prec, is_dagger=is_dagger)
                f_cand = format_mean(cand, is_cand=True)
                f_time = format_mean(time)

                # Format standard deviations
                f_rec_std = format_std(rec_std)
                f_prec_std = format_std(prec_std)
                f_cand_std = format_std(cand_std, is_cand=True)
                f_time_std = format_std(time_std)

                # Append to both rows
                mean_row_str += f" & {f_rec} & {f_prec} & {f_cand} & {f_time}"
                std_row_str += f" & {f_rec_std} & {f_prec_std} & {f_cand_std} & {f_time_std}"

            # Combine mean and std rows
            latex += mean_row_str + " \\\\\n"
            latex += std_row_str + " \\\\\n"

        # Add midrule between methods (except for the last one)
        if method != METHODS[-1]:
            latex += "\\midrule"

    latex += r"""
\hline
\end{tabular}
}
\end{table*}
"""
    return latex

if __name__ == "__main__":
    # Execute the processing
    data = load_data()
    latex_output = generate_latex(data)

    # Save to file
    with open("updated_table.tex", "w") as f:
        f.write(latex_output)

    print("Table generated successfully and saved to 'updated_table.tex'.")