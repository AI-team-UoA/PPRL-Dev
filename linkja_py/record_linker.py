from bloom_filter_encoder import encode_string_to_bloom
from matcher import dice_coefficient
from pseudonymizer import pseudonymize

def compare_records(fields1, fields2, salt="", threshold=0.85, ngram=5):
    if len(fields1) != len(fields2):
        return False, 0.0

    scores = []
    for f1, f2 in zip(fields1, fields2):
        p1 = pseudonymize(f1, salt)
        p2 = pseudonymize(f2, salt)
        bf1 = encode_string_to_bloom(p1, k=ngram)
        bf2 = encode_string_to_bloom(p2, k=ngram)
        scores.append(dice_coefficient(bf1, bf2))
    
    avg_score = sum(scores) / len(scores)
    return avg_score >= threshold, avg_score
