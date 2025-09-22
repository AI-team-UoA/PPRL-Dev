import argparse
from .utils import load_csv_fields
from .record_linker import compare_records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--fields", nargs='+', required=True, help="Fields to match")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--salt", type=str, default="")
    args = parser.parse_args()

    data1 = load_csv_fields(args.file1, args.fields)
    data2 = load_csv_fields(args.file2, args.fields)

    for rec1 in data1:
        for rec2 in data2:
            match, score = compare_records(rec1, rec2, salt=args.salt, threshold=args.threshold, ngram=args.ngram)
            if match:
                print(f"Match: {rec1} <--> {rec2} | Score: {score:.2f}")

if __name__ == "__main__":
    main()
