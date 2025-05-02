# File: data_loader.py

import os
import numpy as np
import wfdb

def load_ecg_windows(
    data_dir="mitdb",
    window_size=250,
    stride=125
):
    os.makedirs(data_dir, exist_ok=True)
    wfdb.dl_database('mitdb', dl_dir=data_dir)
    records = wfdb.get_record_list('mitdb')

    all_windows = []
    all_labels  = []

    for rec in records:
        path = os.path.join(data_dir, rec)
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')

        sig = record.p_signal[:, 0]  # Lead I
        
        # Normalize to [0, 1]
        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min > 1e-6:  # Avoid division by zero
            sig = (sig - sig_min) / (sig_max - sig_min)
        else:
            sig = sig - sig_min  # Handle flatline signals

        arr_idx = {
            int(idx)
            for idx, sym in zip(ann.sample, ann.symbol)
            if sym != 'N'
        }

        for start in range(0, len(sig) - window_size + 1, stride):
            end = start + window_size
            window = sig[start:end]
            label = any(start <= idx < end for idx in arr_idx)
            all_windows.append(window)
            all_labels.append(label)

    X = np.stack(all_windows).astype(np.float32)
    is_outlier = np.array(all_labels, dtype=bool)
    return X, is_outlier

# Optional helper to cache windows to disk for faster reuse
def cache_ecg(
    data_dir="mitdb",
    window_size=250,
    stride=125,
    cache_dir="data/cache"
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_X = os.path.join(cache_dir, "X_ecg.npy")
    cache_y = os.path.join(cache_dir, "is_outlier_ecg.npy")

    if not os.path.exists(cache_X):
        X, is_out = load_ecg_windows(data_dir, window_size, stride)
        np.save(cache_X, X)
        np.save(cache_y, is_out)
        print(f"✅ Cached ECG windows: {X.shape}, outliers: {is_out.sum()} / {len(is_out)}")
    else:
        print(f"✅ Found cached ECG at {cache_dir}")

if __name__ == "__main__":
    # Run this once to download, process, and cache the windows
    cache_ecg()
