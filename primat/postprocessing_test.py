from postprocessing import classify_pairs, greedy_one_to_one, evaluate_linkage

# Assume sim_scores is obtained from similarity computation between blocked candidate pairs
sim_scores = [
    (0,1,0.85), (0,2,0.65), (1,3,0.9), (2,3,0.4)
]

# 1. Filter by threshold
candidates = classify_pairs(sim_scores, threshold=0.8)

# 2. Enforce one-to-one matches (optional)
final_matches = greedy_one_to_one(candidates)
predicted = [(i,j) for i,j,_ in final_matches]

# 3. Ground truth known duplicates
truth = [(0,1), (1,3)]

# 4. Evaluate
metrics = evaluate_linkage(predicted, truth, total_possible_pairs=100)
print(metrics)
