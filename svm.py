"""
Support Vector Machine (SVM) for Binary Classification
Supports: Linear, Polynomial, RBF (Gaussian), Sigmoid kernels

Uses SMO (Sequential Minimal Optimization) algorithm to solve the
dual quadratic programming problem.

Reference: Platt, J. (1998). "Sequential Minimal Optimization"
"""

import numpy as np
from collections import Counter


class SVM:
    """
    Binary SVM Classifier using SMO solver.

    Parameters:
        C:       Regularization (soft margin). Larger → less tolerance for misclassification.
        kernel:  'linear', 'poly', 'rbf', 'sigmoid'
        gamma:   Kernel coefficient for rbf/poly/sigmoid. 'auto' = 1/n_features.
        degree:  Degree for polynomial kernel.
        coef0:   Independent term in poly/sigmoid kernels.
        tol:     Tolerance for KKT violation.
        max_iter: Maximum SMO passes.
    """

    def __init__(self, C=1.0, kernel='linear', gamma='auto', degree=3,
                 coef0=0.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    # ── Kernel functions ──────────────────────────────────────────

    def _kernel_fn(self, x1, x2):
        """Compute kernel between two vectors."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self._gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self._gamma * np.dot(diff, diff))
        elif self.kernel == 'sigmoid':
            return np.tanh(self._gamma * np.dot(x1, x2) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _kernel_matrix(self, X):
        """Precompute full kernel matrix K[i,j] = K(X[i], X[j])."""
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._kernel_fn(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    # ── SMO solver ────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train SVM using SMO algorithm.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) labels, must be +1 or -1
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Convert labels to +1/-1 if needed
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"Binary classification requires 2 classes, got {len(classes)}")
        self.classes_ = classes
        if not (set(classes) == {-1, 1} or set(classes) == {-1.0, 1.0}):
            # Map to -1/+1
            self._label_map = {classes[0]: -1.0, classes[1]: 1.0}
            self._label_inv = {-1.0: classes[0], 1.0: classes[1]}
            y = np.array([self._label_map[yi] for yi in y])
        else:
            self._label_map = None

        n, d = X.shape
        self._gamma = 1.0 / d if self.gamma == 'auto' else self.gamma

        # Initialize
        alpha = np.zeros(n)
        b = 0.0
        K = self._kernel_matrix(X)

        # Error cache
        E = np.zeros(n)
        for i in range(n):
            E[i] = self._decision_raw(K, alpha, y, b, i) - y[i]

        passes = 0
        while passes < self.max_iter:
            num_changed = 0

            for i in range(n):
                Ei = E[i]
                # Check KKT violation
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                        (y[i] * Ei > self.tol and alpha[i] > 0):

                    # Select j ≠ i (heuristic: max |Ei - Ej|)
                    j = self._select_j(i, Ei, E, alpha, n)
                    Ej = E[j]

                    ai_old, aj_old = alpha[i], alpha[j]

                    # Compute bounds L, H
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if abs(L - H) < 1e-10:
                        continue

                    # Second derivative
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha[i] += y[i] * y[j] * (aj_old - alpha[j])

                    # Update bias
                    b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i, i] \
                         - y[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i, j] \
                         - y[j] * (alpha[j] - aj_old) * K[j, j]

                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    # Update error cache
                    for k in range(n):
                        E[k] = self._decision_raw(K, alpha, y, b, k) - y[k]

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # Store support vectors
        sv_idx = alpha > 1e-7
        self.alpha_ = alpha[sv_idx]
        self.sv_ = X[sv_idx]
        self.sv_y_ = y[sv_idx]
        self.b_ = b
        self.n_support_ = np.sum(sv_idx)

        # For linear kernel, compute weight vector w
        if self.kernel == 'linear':
            self.w_ = np.sum((self.alpha_ * self.sv_y_)[:, None] * self.sv_, axis=0)

        return self

    def _decision_raw(self, K, alpha, y, b, i):
        """Raw decision value for sample i using precomputed kernel matrix."""
        return np.sum(alpha * y * K[:, i]) + b

    def _select_j(self, i, Ei, E, alpha, n):
        """Select second index j using max |Ei - Ej| heuristic."""
        max_delta = -1
        j = i
        # Prefer non-bound examples
        non_bound = np.where((alpha > 1e-7) & (alpha < self.C - 1e-7))[0]
        candidates = non_bound if len(non_bound) > 1 else range(n)

        for k in candidates:
            if k == i:
                continue
            delta = abs(Ei - E[k])
            if delta > max_delta:
                max_delta = delta
                j = k
        if j == i:
            # Random fallback
            j = i
            while j == i:
                j = np.random.randint(0, n)
        return j

    # ── Prediction ────────────────────────────────────────────────

    def decision_function(self, X):
        """Compute raw decision values (signed distance to hyperplane)."""
        X = np.array(X, dtype=np.float64)
        n = X.shape[0]
        dec = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(len(self.alpha_)):
                s += self.alpha_[j] * self.sv_y_[j] * self._kernel_fn(self.sv_[j], X[i])
            dec[i] = s + self.b_
        return dec

    def predict(self, X):
        """Predict class labels."""
        dec = self.decision_function(X)
        preds = np.sign(dec)
        preds[preds == 0] = 1.0  # tie-break
        if self._label_map is not None:
            return np.array([self._label_inv[p] for p in preds])
        return preds

    def score(self, X, y):
        """Classification accuracy."""
        return np.mean(self.predict(X) == y)


# ============================================================================
# Demo and Testing
# ============================================================================

def make_linear_data(n=200, seed=42):
    """Linearly separable 2D data."""
    rng = np.random.RandomState(seed)
    X0 = rng.randn(n // 2, 2) + np.array([2, 2])
    X1 = rng.randn(n // 2, 2) + np.array([-2, -2])
    X = np.vstack([X0, X1])
    y = np.array([1] * (n // 2) + [-1] * (n // 2), dtype=float)
    return X, y


def make_circle_data(n=300, seed=42):
    """Non-linearly separable circular data."""
    rng = np.random.RandomState(seed)
    r1 = rng.uniform(0, 1.5, n // 2)
    t1 = rng.uniform(0, 2 * np.pi, n // 2)
    X0 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])

    r2 = rng.uniform(2.5, 4.0, n // 2)
    t2 = rng.uniform(0, 2 * np.pi, n // 2)
    X1 = np.column_stack([r2 * np.cos(t2), r2 * np.sin(t2)])

    X = np.vstack([X0, X1])
    y = np.array([1] * (n // 2) + [-1] * (n // 2), dtype=float)
    return X, y


def make_xor_data(n=200, seed=42):
    """XOR-pattern data (needs kernel)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = np.where(X[:, 0] * X[:, 1] > 0, 1.0, -1.0)
    return X, y


def train_test_split(X, y, test_ratio=0.3, seed=42):
    """Simple train/test split."""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


def run_kernel(name, X, y, kernel, **kwargs):
    """Run a kernel on a dataset and print results."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)

    svm = SVM(C=1.0, kernel=kernel, **kwargs)
    svm.fit(X_tr, y_tr)

    train_acc = svm.score(X_tr, y_tr)
    test_acc = svm.score(X_te, y_te)

    print(f"  {name:30s}  │  kernel={kernel:8s}  │"
          f"  SV={svm.n_support_:>4d}  │"
          f"  Train={train_acc:.3f}  Test={test_acc:.3f}")
    return svm


def main():
    print("=" * 85)
    print("  SVM Binary Classifier — Linear and Kernel")
    print("=" * 85)

    # ── Dataset 1: Linearly separable ──
    print("\n  Dataset 1: Linearly Separable (2D Gaussians)")
    print("  " + "─" * 80)
    X_lin, y_lin = make_linear_data()
    run_kernel("Linear data + linear", X_lin, y_lin, 'linear')
    # run_kernel("Linear data + RBF",        X_lin, y_lin, 'rbf', gamma=0.5)
    # run_kernel("Linear data + Poly(d=2)",  X_lin, y_lin, 'poly', degree=2, gamma=1.0)

    # ── Dataset 2: Circular (non-linear) ──
    # print("\n  Dataset 2: Circular (Non-Linear)")
    # print("  " + "─" * 80)
    # X_circ, y_circ = make_circle_data()
    # run_kernel("Circle data + linear",     X_circ, y_circ, 'linear')
    # run_kernel("Circle data + RBF(γ=1)",   X_circ, y_circ, 'rbf', gamma=1.0)
    # run_kernel("Circle data + RBF(γ=0.5)", X_circ, y_circ, 'rbf', gamma=0.5)
    # run_kernel("Circle data + Poly(d=2)",  X_circ, y_circ, 'poly', degree=2, gamma=1.0)
    #
    # # ── Dataset 3: XOR pattern ──
    # print("\n  Dataset 3: XOR Pattern")
    # print("  " + "─" * 80)
    # X_xor, y_xor = make_xor_data()
    # run_kernel("XOR data + linear",        X_xor, y_xor, 'linear')
    # run_kernel("XOR data + RBF(γ=1)",      X_xor, y_xor, 'rbf', gamma=1.0)
    # run_kernel("XOR data + Poly(d=2)",     X_xor, y_xor, 'poly', degree=2, gamma=1.0)
    # run_kernel("XOR data + Sigmoid",       X_xor, y_xor, 'sigmoid', gamma=1.0, coef0=0.0)
    #
    # # ── Label mapping test (0/1 labels) ──
    # print("\n  Dataset 4: Label Mapping (0/1 → -1/+1)")
    # print("  " + "─" * 80)
    # X_01, y_01 = make_linear_data()
    # y_01 = np.where(y_01 == 1, 1, 0)  # convert to 0/1
    # run_kernel("0/1 labels + linear",      X_01, y_01, 'linear')
    #
    # # ── Linear kernel: show weight vector ──
    # print("\n  Linear Kernel Weight Vector:")
    # print("  " + "─" * 80)
    # svm_lin = SVM(C=1.0, kernel='linear')
    # svm_lin.fit(*make_linear_data())
    # print(f"    w = {svm_lin.w_}")
    # print(f"    b = {svm_lin.b_:.4f}")
    # print(f"    Decision: f(x) = w·x + b = {svm_lin.w_[0]:.4f}·x₁ + {svm_lin.w_[1]:.4f}·x₂ + {svm_lin.b_:.4f}")
    #
    # print("\n" + "=" * 85)
    # print("  All tests complete!")
    # print("=" * 85)


if __name__ == "__main__":
    main()
