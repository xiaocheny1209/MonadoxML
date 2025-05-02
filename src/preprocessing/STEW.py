import os
import numpy as np

def load_ratings(rating_file):
    """
    Loads the workload ratings for each subject from ratings.txt.
    Returns a dict: {subject_id: {"rest": int, "task": int}}
    """
    ratings = {}
    with open(rating_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                sub_id, rest, test = map(int, parts)
                ratings[f"{sub_id:02d}"] = {"rest": rest, "task": test}
    return ratings


def load_stew_data(data_dir="data/STEW", mode="task", threshold=5):
    """
    Loads EEG data and binary labels from the STEW dataset.

    Parameters:
    - mode: "task" for multitasking recordings (hi), "rest" for resting state (lo)
    - threshold: workload rating threshold to define binary classification

    Returns:
    - X: list of (samples x 14) numpy arrays
    - y: list of binary labels (0 = low workload, 1 = high workload)
    """
    assert mode in ("task", "rest")
    ratings = load_ratings(os.path.join(data_dir, "ratings.txt"))
    X = []
    y = []

    suffix = "hi" if mode == "task" else "lo"

    for filename in os.listdir(data_dir):
        if filename.endswith(f"_{suffix}.txt"):
            subject = filename[3:5]  # e.g., "01", "23", etc.
            if subject not in ratings:
                continue  # skip if rating is missing

            try:
                data = np.loadtxt(os.path.join(data_dir, filename))
                if data.shape[1] != 14:
                    continue  # ignore malformed files

                rating = ratings[subject][mode]
                label = int(rating >= threshold)  # 1 = high, 0 = low
                X.append(data)
                y.append(label)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

    return X, y
