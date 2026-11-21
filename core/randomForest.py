import numpy as np

from core.model import Model


class RandomForest(Model):
    """Ensemble of randomized binary decision trees for infection prediction."""

    def __init__(
        self,
        n_estimators=25,
        max_depth=6,
        max_features="sqrt",
        min_samples_split=2,
        seed=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y, optimizer=None, epochs=500):
        del optimizer, epochs
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must match its number of rows")
        if len(X) == 0:
            raise ValueError("Cannot fit a random forest on an empty dataset")

        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = self._rng.integers(0, len(X), len(X))
            self.trees.append(self._build_tree(X[indices], y[indices], 0))
        return self

    def predict(self, X):
        if not hasattr(self, "trees"):
            raise RuntimeError("The model must be fitted before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features used for fitting")
        return np.mean(
            [np.array([self._predict_tree(tree, row) for row in X]) for tree in self.trees],
            axis=0,
        )

    def _build_tree(self, X, y, depth):
        probability = float(y.mean())
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or probability in (0.0, 1.0)
        ):
            return {"probability": probability}

        feature_count = self._feature_count()
        features = self._rng.choice(self.n_features_in_, feature_count, replace=False)
        split = self._best_split(X, y, features)
        if split is None:
            return {"probability": probability}

        feature, threshold = split
        left = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], y[left], depth + 1),
            "right": self._build_tree(X[~left], y[~left], depth + 1),
        }

    def _feature_count(self):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_in_)))
        return min(self.n_features_in_, int(self.max_features))

    @staticmethod
    def _best_split(X, y, features):
        best_score = np.inf
        best_split = None
        for feature in features:
            values = np.unique(X[:, feature])
            for threshold in (values[:-1] + values[1:]) / 2:
                left = X[:, feature] <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                score = (
                    left.sum() * RandomForest._gini(y[left])
                    + right.sum() * RandomForest._gini(y[right])
                ) / len(y)
                if score < best_score:
                    best_score = score
                    best_split = (feature, threshold)
        return best_split

    @staticmethod
    def _gini(y):
        probability = y.mean()
        return 2 * probability * (1 - probability)

    @staticmethod
    def _predict_tree(tree, row):
        while "probability" not in tree:
            branch = "left" if row[tree["feature"]] <= tree["threshold"] else "right"
            tree = tree[branch]
        return tree["probability"]
