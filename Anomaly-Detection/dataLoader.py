import os
import csv

# Root dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "Data/OMTAD/WestGrid")
OUTPUT_TYPICAL = os.path.join(BASE_DIR, "Data/typical.csv")
OUTPUT_ANOMALOUS = os.path.join(BASE_DIR, "Data/anomalous.csv")

TYPICAL_TYPES = {"cargo", "passenger", "tanker"}
ANOMALOUS_TYPES = {"fishing"}


def find_mpf_files(root_dir):
    """
    Walk through root_dir and yield (filepath, vessel_type).
    Vessel type is taken from the folder name (cargo/fishing/passenger/tanker).
    """
    for dirpath, _, filenames in os.walk(root_dir):
        parts = dirpath.split(os.sep)
        if len(parts) >= 2:
            vessel_type = parts[-2].lower()  # cargo, fishing, passenger, tanker
        else:
            vessel_type = None

        for fname in filenames:
            if fname.lower().startswith("mpf_") and fname.lower().endswith(".csv"):
                yield os.path.join(dirpath, fname), vessel_type


def read_mpf(filepath):
    """
    Reads an MPF CSV, returns rows (excluding END separators) and header.
    """
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row:
                continue
            if row[0].strip().upper() == "END":
                continue
            rows.append(row)
    return header, rows


def combine_all(root_dir, typical_file, anomalous_file):
    # Ensure old output files are deleted
    for f in (typical_file, anomalous_file):
        if os.path.exists(f):
            os.remove(f)

    mpf_files = list(find_mpf_files(root_dir))
    if not mpf_files:
        print("No MPF files found in", root_dir)
        return

    # Get header from first file
    first_file, _ = mpf_files[0]
    header, _ = read_mpf(first_file)

    with open(typical_file, "w", newline="", encoding="utf-8") as ft, \
         open(anomalous_file, "w", newline="", encoding="utf-8") as fa:
        writer_typical = csv.writer(ft)
        writer_anomalous = csv.writer(fa)

        # Write header to both
        writer_typical.writerow(header)
        writer_anomalous.writerow(header)

        typical_count = 0
        anomalous_count = 0

        for fp, vessel_type in mpf_files:
            _, rows = read_mpf(fp)

            if vessel_type in TYPICAL_TYPES:
                writer_typical.writerows(rows)
                typical_count += len(rows)

            elif vessel_type in ANOMALOUS_TYPES:
                writer_anomalous.writerows(rows)
                anomalous_count += len(rows)

    print(f"Wrote {typical_count} rows to {typical_file}")
    print(f"Wrote {anomalous_count} rows to {anomalous_file}")


if __name__ == "__main__":
    combine_all(DATA_ROOT, OUTPUT_TYPICAL, OUTPUT_ANOMALOUS)
