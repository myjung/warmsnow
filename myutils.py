import csv

def read_csv(file:str)->list[dict]:
    """
    data fields
    Name: Text ID
    CHT: Original traditional Chinese 
    KOR: Translated Korean
    """
    with open(file, encoding="utf-8") as f:
        first_8_letter = f.read(8)
        if first_8_letter.startswith('\ufeff'):
            f.seek(3)
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        return list(reader)