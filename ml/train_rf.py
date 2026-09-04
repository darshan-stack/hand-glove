"""Train the leakage-safe Random Forest baseline.

Usage:
    python -m ml.train_rf --csv data/raw/gestures.csv
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from ml.dataset import load_csv
from preprocessing.pipeline import fit_calibration, add_motion_features, make_windows
from ml.features import build_feature_matrix


def build_windows_for_df(df, window=32, stride=8):
    all_x, all_y = [], []
    for user_id, user_df in df.groupby("user_id", sort=False):
        # Calibration is estimated only from this user's first neutral samples
        # if available; otherwise median/IQR over that user's recording is used
        # for a reproducible baseline. Production enrollment should be explicit.
        raw = user_df[["flex1","flex2","flex3","flex4","flex5","ax","ay","az","gx","gy","gz"]].to_numpy(np.float32)
        cal = fit_calibration(raw)
        x = add_motion_features(cal.transform(raw))
        y = user_df["gesture"].astype(str).to_numpy()
        wx, wy = make_windows(x, y, window=window, stride=stride)
        if len(wx):
            all_x.append(wx); all_y.append(wy)
    if not all_x:
        raise ValueError("No complete gesture windows were created")
    return np.concatenate(all_x), np.concatenate(all_y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="artifacts/rf")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()

    df = load_csv(args.csv)
    users = sorted(df.user_id.astype(str).unique())
    if len(users) < 2:
        raise ValueError("Need at least 2 users for a user-independent evaluation")
    rng = np.random.default_rng(42)
    rng.shuffle(users)
    n_test = max(1, int(round(len(users) * 0.2)))
    test_users = set(users[:n_test])
    train_df = df[~df.user_id.astype(str).isin(test_users)].copy()
    test_df = df[df.user_id.astype(str).isin(test_users)].copy()

    x_train, y_train = build_windows_for_df(train_df, args.window, args.stride)
    x_test, y_test = build_windows_for_df(test_df, args.window, args.stride)
    le = LabelEncoder().fit(y_train)
    # Test gestures unseen during training are reported as unknown rather than
    # being silently encoded as a known class.
    known = np.isin(y_test, le.classes_)
    if not known.any():
        raise ValueError("No test gesture classes overlap training classes")
    x_test, y_test = x_test[known], y_test[known]
    y_train_i, y_test_i = le.transform(y_train), le.transform(y_test)

    Xtr, Xte = build_feature_matrix(x_train), build_feature_matrix(x_test)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    clf.fit(Xtr, y_train_i)
    pred = clf.predict(Xte)

    metrics = {
        "accuracy": float(accuracy_score(y_test_i, pred)),
        "classification_report": classification_report(y_test_i, pred, target_names=le.classes_, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test_i, pred).tolist(),
        "test_users": sorted(test_users),
        "n_train_windows": int(len(Xtr)),
        "n_test_windows": int(len(Xte)),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le, "feature_version": 1}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"accuracy": metrics["accuracy"], "test_users": metrics["test_users"]}, indent=2))


if __name__ == "__main__":
    main()
