import numpy as np

def generate_1d(n_samples=10000, modes=[(0, 1), (5, 2)], outlier_frac=0.1, seed=42):
    """
    Generate 1D signal data as a mixture of Gaussians with injected uniform outliers.
    Returns:
      X: np.ndarray of shape (n_samples, 1)
      is_outlier: np.ndarray of booleans of shape (n_samples,)
    """
    rng = np.random.RandomState(seed)
    n_modes = len(modes)
    # samples per mode
    samples_per = n_samples // n_modes
    X = []
    for mu, sigma in modes:
        X.append(rng.normal(loc=mu, scale=sigma, size=samples_per))
    X = np.concatenate(X)
    # Pad if needed
    if X.shape[0] < n_samples:
        extra = rng.normal(loc=modes[0][0], scale=modes[0][1], size=(n_samples - X.shape[0],))
        X = np.concatenate([X, extra])
    # Inject outliers
    is_outlier = np.zeros(n_samples, dtype=bool)
    n_out = int(outlier_frac * n_samples)
    out_idx = rng.choice(n_samples, size=n_out, replace=False)
    X[out_idx] = rng.uniform(X.min() - 5, X.max() + 5, size=n_out)
    is_outlier[out_idx] = True
    return X.reshape(-1, 1).astype(np.float32), is_outlier