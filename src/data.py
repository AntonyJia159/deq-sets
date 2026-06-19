"""Synthetic set tasks for probing set-DEQ properties.

Layer-1 task: Gaussian-mixture sets. Each sample is a set of points drawn from
k well-separated isotropic Gaussians; the label is k (an invariant target).
The task genuinely needs iterative mutual refinement (a point's cluster identity
depends on the global configuration), so it exercises the equilibrium dynamics
rather than a single feed-forward pass.
"""

import torch


def _sample_separated_centers(k, d, sep, std, generator, max_tries=1000):
    """Sample k centers with pairwise distance >= sep * std (rejection sampling)."""
    scale = sep * std * max(1, k) ** (1.0 / max(d, 1))
    for _ in range(max_tries):
        centers = (torch.rand(k, d, generator=generator) - 0.5) * 2 * scale
        if k == 1:
            return centers
        dists = torch.cdist(centers, centers)
        dists = dists + torch.eye(k) * 1e9  # ignore self-distance
        if dists.min() >= sep * std:
            return centers
    # Fall back to the last draw if we never satisfied the constraint.
    return centers


def sample_gmm_set(k, n_points, d=2, sep=4.0, std=1.0, generator=None):
    """Return a single set X of shape (n_points, d) drawn from k Gaussians."""
    centers = _sample_separated_centers(k, d, sep, std, generator)
    # Guarantee every cluster gets at least one point (when N >= k) so the label
    # k stays faithful to the actual content, at any cardinality.
    if n_points >= k:
        base = torch.arange(k)
        rest = torch.randint(0, k, (n_points - k,), generator=generator)
        assign = torch.cat([base, rest])
        assign = assign[torch.randperm(n_points, generator=generator)]
    else:
        assign = torch.randint(0, k, (n_points,), generator=generator)
    noise = torch.randn(n_points, d, generator=generator) * std
    X = centers[assign] + noise
    return X, assign


class GMMSetDataset:
    """In-memory dataset of Gaussian-mixture sets with cluster-count labels.

    Labels are k - k_min so they index a classifier output of size
    (k_max - k_min + 1).
    """

    def __init__(self, n_samples, k_range=(1, 5), n_points=30, d=2,
                 sep=4.0, std=1.0, seed=0):
        self.k_min, self.k_max = k_range
        self.n_points = n_points
        self.d = d
        self.n_classes = self.k_max - self.k_min + 1
        g = torch.Generator().manual_seed(seed)
        self.X, self.y, self.assign = [], [], []
        for _ in range(n_samples):
            k = int(torch.randint(self.k_min, self.k_max + 1, (1,), generator=g))
            X, a = sample_gmm_set(k, n_points, d, sep, std, g)
            self.X.append(X)
            self.y.append(k - self.k_min)
            self.assign.append(a)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)

    def batch(self, idx):
        """Stack samples at the given indices into (B, N, d), (B,) tensors.

        Assumes a fixed n_points across the dataset. Variable-cardinality
        batching (padding + masking) is deferred to a later layer.
        """
        X = torch.stack([self.X[i] for i in idx])
        y = self.y[idx]
        return X, y

    def iter_batches(self, batch_size, shuffle=True, seed=0):
        n = len(self)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) \
            if shuffle else torch.arange(n)
        for s in range(0, n, batch_size):
            yield self.batch(order[s:s + batch_size])
