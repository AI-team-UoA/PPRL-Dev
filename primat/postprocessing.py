from typing import List, Tuple, Dict

def classify_pairs(similarities: List[Tuple[int,int,float]], threshold: float) -> List[Tuple[int,int,float]]:
    """
    Select pairs with similarity >= threshold.
    similarities: list of (id1, id2, score)
    """
    return [(i, j, s) for (i, j, s) in similarities if s >= threshold]

def greedy_one_to_one(matches: List[Tuple[int,int,float]]) -> List[Tuple[int,int,float]]:
    """
    Enforce one-to-one matches by selecting highest-score edges first.
    matches: list of (i, j, score)
    """
    sorted_pairs = sorted(matches, key=lambda x: x[2], reverse=True)
    selected = []
    used_i, used_j = set(), set()
    for i, j, s in sorted_pairs:
        if i not in used_i and j not in used_j:
            selected.append((i, j, s))
            used_i.add(i)
            used_j.add(j)
    return selected

def evaluate_linkage(
    predicted: List[Tuple[int,int]],
    ground_truth: List[Tuple[int,int]],
    total_possible_pairs: int = None
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1-score, and Reduction Ratio.
    ground_truth and predicted are sets/lists of matching index tuples.
    total_possible_pairs: number of candidate comparisons avoided (blocking reduction).
    """
    gt_set = set(ground_truth)
    pred_set = set(predicted)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    rr = None
    if total_possible_pairs is not None:
        compared = len(pred_set) + fp  # or actual comparisons performed
        rr = 1.0 - (compared / total_possible_pairs)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reduction_ratio": rr,
    }
