from __future__ import annotations

import re
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LimeTextExplainer:

    def __init__(
        self,
        predict_fn: Callable[[List[str]], np.ndarray],
        kernel_width: float = 25.0,
        num_samples: int = 5000,
        feature_selection: str = "auto",
        random_seed: int = 42
    ):
        """
        Args:
            predict_fn: Function that takes list of texts and returns phishing probabilities
            kernel_width: Width of exponential kernel for locality weights
            num_samples: Number of perturbed samples to generate
            feature_selection: Method for feature selection ("auto", "none", "lasso_path")
        """
        self.predict_fn = predict_fn
        self.kernel_width = kernel_width
        self.num_samples = num_samples
        self.feature_selection = feature_selection
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def tokenize_words(self, text: str) -> List[str]:
        """Split text into interpretable words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _generate_perturbations(
        self, tokens: List[str]
    ) -> Tuple[List[List[int]], np.ndarray]:
        """Generate binary masks for token perturbations."""
        n_tokens = len(tokens)
        perturbations = []
        distances = []

        for _ in range(self.num_samples):
            # Binary mask: 1 means keep token, 0 means remove
            mask = np.random.binomial(1, 0.5, size=n_tokens)
            perturbations.append(mask)

            # Compute Hamming distance from original (all 1's)
            distance = np.sum(1 - mask)
            distances.append(distance)

        return perturbations, np.array(distances)

    def _apply_perturbation(self, tokens: List[str], mask: List[int]) -> str:
        """Apply binary mask to tokens to create perturbed text."""
        kept_tokens = [token for token, keep in zip(tokens, mask) if keep == 1]
        return " ".join(kept_tokens)

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances ** 2) / (self.kernel_width ** 2))

    def _fit_local_surrogate(
        self,
        perturbations: List[List[int]],
        weights: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fit sparse linear model to local neighborhood."""
        X = np.array(perturbations)
        y = labels

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit ridge regression with L2 regularization
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y, sample_weight=weights)

        # Get coefficients and intercept
        coefficients = ridge.coef_
        intercept = ridge.intercept_

        return coefficients, intercept

    def explain_instance(
        self,
        text: str,
        num_features: int = 10,
        positive_only: bool = False
    ) -> Dict[str, Any]:
        """
        Explain a single text instance.

        Args:
            text: Input text to explain
            num_features: Number of top features to return
            positive_only: If True, only return positive contributions

        Returns:
            Dictionary containing explanation results
        """
        # Tokenize text
        tokens = self.tokenize_words(text)
        if not tokens:
            return {"error": "Empty text", "tokens": [], "contributions": []}

        # Get original prediction
        original_proba = float(self.predict_fn([text])[0])

        # Generate perturbations
        perturbations, distances = self._generate_perturbations(tokens)

        # Create perturbed texts
        perturbed_texts = []
        valid_indices = []

        for i, mask in enumerate(perturbations):
            # Skip empty texts (no tokens kept)
            if np.sum(mask) == 0:
                continue

            perturbed_text = self._apply_perturbation(tokens, mask)
            perturbed_texts.append(perturbed_text)
            valid_indices.append(i)

        if not perturbed_texts:
            return {"error": "No valid perturbations", "tokens": tokens, "contributions": []}

        # Get predictions for perturbed texts
        perturbed_probas = self.predict_fn(perturbed_texts)

        # Filter distances and perturbations to match valid samples
        valid_distances = distances[valid_indices]
        valid_perturbations = [perturbations[i] for i in valid_indices]

        # Compute weights
        weights = self._compute_weights(valid_distances)

        # Fit local surrogate model
        coefficients, intercept = self._fit_local_surrogate(
            valid_perturbations, weights, perturbed_probas
        )

        # Create token contributions
        contributions = []
        for i, token in enumerate(tokens):
            if i < len(coefficients):
                contributions.append({
                    "token": token,
                    "weight": float(coefficients[i]),
                    "position": i
                })

        # Sort by absolute weight
        contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)

        # Filter if positive_only
        if positive_only:
            contributions = [c for c in contributions if c["weight"] > 0]

        # Get top features
        top_features = contributions[:num_features]
        top_positive = [c for c in top_features if c["weight"] > 0]
        top_negative = [c for c in top_features if c["weight"] < 0]

        return {
            "original_text": text,
            "tokens": tokens,
            "original_probability": original_proba,
            "predicted_class": "phishing" if original_proba >= 0.5 else "safe",
            "num_features_considered": len(contributions),
            "top_positive_tokens": top_positive,
            "top_negative_tokens": top_negative,
            # Keep top 2x
            "all_contributions": contributions[:num_features * 2],
            "surrogate_intercept": float(intercept)
        }
