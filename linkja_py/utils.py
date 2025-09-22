import csv

def load_csv_fields(file_path, field_names):
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        return [[row[field] for field in field_names] for row in reader]
