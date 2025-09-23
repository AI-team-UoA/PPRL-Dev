from utils import load_csv_fields
from record_linker import compare_records

def main(file1: str,
        file2 :str,
        fields: list,
        threshold: float = 0.85,
        ngram: int = 5,
        salt: str = ""):
    
    data1 = load_csv_fields(file1, fields)
    data2 = load_csv_fields(file2, fields)

    for rec1 in data1:
        for rec2 in data2:
            match, score = compare_records(rec1, rec2, salt=salt, threshold=threshold, ngram=ngram)
            print(f"Match: {rec1} <--> {rec2} | Score: {score:.2f}")

if __name__ == "__main__":
    path = "../data/test"
    main(file1 = f"{path}/test1.csv",
         file2 = f"{path}/test2.csv",
         fields=['name', 'email', 'phone'],
         ngram=5,
         salt='test')


