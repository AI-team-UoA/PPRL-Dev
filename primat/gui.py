import streamlit as st
import pandas as pd
from bloom_filter import BloomFilterEncoder
from blocking import LSHBlocker
from postprocessing import classify_pairs, greedy_one_to_one, evaluate_linkage
from bitarray import bitarray
import itertools

# ------------------ UI: Streamlit App ------------------
st.set_page_config(page_title="Privacy-Preserving Record Linkage", layout="wide")
st.title("🔐 Privacy-Preserving Record Linkage (PRIMAT-inspired)")

# ------------------ Upload & Clean Data ------------------
st.header("1. Upload and Preview Datasets")
left_file = st.file_uploader("Upload Dataset A (CSV)", type="csv")
right_file = st.file_uploader("Upload Dataset B (CSV)", type="csv")

if left_file and right_file:
    df_a = pd.read_csv(left_file).fillna("")
    df_b = pd.read_csv(right_file).fillna("")

    st.write("Preview Dataset A:", df_a.head())
    st.write("Preview Dataset B:", df_b.head())

    # ------------------ Parameters ------------------
    st.header("2. Configure Linkage Parameters")
    fields = st.multiselect("Select Fields for Encoding", df_a.columns.tolist(), default=df_a.columns.tolist()[:2])
    size = st.slider("Bloom Filter Size (m)", 128, 2048, 512, step=128)
    k = st.slider("Number of Hashes (k)", 1, 10, 5)
    q = st.slider("q-gram Length (q)", 1, 5, 2)
    psi = st.slider("LSH Key Length (ψ)", 12, 64, 36)
    lmbda = st.slider("LSH Keys per Record (λ)", 1, 10, 3)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.85)
    one_to_one = st.checkbox("Enforce one-to-one matching", value=True)

    # ------------------ Encode ------------------
    st.header("3. Encode Records with Bloom Filters")
    encoder = BloomFilterEncoder(size=size, num_hashes=k, q=q)

    def record_to_string(row):
        return "#".join(str(row[f]) for f in fields)

    bloom_a = [encoder.encode(record_to_string(row)) for _, row in df_a.iterrows()]
    bloom_b = [encoder.encode(record_to_string(row)) for _, row in df_b.iterrows()]

    # ------------------ Block ------------------
    st.header("4. Blocking with LSH")
    blocker = LSHBlocker(psi=psi, lmbda=lmbda)
    blocker.fit(bloom_a + bloom_b)

    blocks_a = blocker.block_all(bloom_a)
    blocks_b = blocker.block_all(bloom_b)

    # ------------------ Candidate Generation ------------------
    st.header("5. Similarity Comparison and Matching")
    candidates = set()
    for key in blocks_a:
        ids_a = blocks_a.get(key, [])
        ids_b = blocks_b.get(key, [])
        for i, j in itertools.product(ids_a, ids_b):
            candidates.add((i, j))

    def dice_sim(b1: bitarray, b2: bitarray) -> float:
        inter = (b1 & b2).count()
        return (2 * inter) / (b1.count() + b2.count()) if (b1.count() + b2.count()) else 0.0

    sim_scores = [(i, j, dice_sim(bloom_a[i], bloom_b[j])) for i, j in candidates]
    filtered = classify_pairs(sim_scores, threshold)
    final_matches = greedy_one_to_one(filtered) if one_to_one else filtered

    # ------------------ Output ------------------
    st.subheader("Matching Results")
    results = pd.DataFrame([
        {"A_ID": i, "B_ID": j, "Similarity": round(s, 3)} for i, j, s in final_matches
    ])
    st.dataframe(results)

    # ------------------ Optional Evaluation ------------------
    st.header("6. Evaluation (Optional)")
    truth_upload = st.file_uploader("Upload Ground Truth File (CSV with A_ID,B_ID columns)", type="csv")
    if truth_upload:
        truth_df = pd.read_csv(truth_upload)
        truth_pairs = list(map(tuple, truth_df[['A_ID', 'B_ID']].values))
        predicted_pairs = [(i, j) for i, j, _ in final_matches]
        total = len(df_a) * len(df_b)
        metrics = evaluate_linkage(predicted_pairs, truth_pairs, total_possible_pairs=total)
        st.json(metrics)

    st.success(f"Linkage completed: {len(final_matches)} matches found.")