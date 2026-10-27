"""Typed, identified errors. Every expected user/data/protocol error maps to exit code 2."""

from __future__ import annotations


class RecError(Exception):
    """Expected, identified failure (malformed input, corrupt data, protocol/phase violation)."""

    exit_code = 2

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class DataError(RecError):
    """Invalid or missing dataset input."""


class SplitError(RecError):
    """Temporal split cannot be computed (e.g. UNSPLITTABLE_CUTOFFS)."""


class ArtifactError(RecError):
    """Missing/corrupt persisted artifacts or identity/protocol mismatch."""


class PhaseError(RecError):
    """Operation prohibited in the current workflow (e.g. evaluating the sealed TEST split with `evaluate`)."""


class EvaluationError(RecError):
    """Evaluation cannot be performed (e.g. NO_EVALUABLE_USERS, invalid ranking)."""


class ModelIntegrityError(RecError):
    """Model fitting failure, nonfinite/invalid factors or scores (unexpected -> exit code 1)."""

    exit_code = 1

class BundleError(ArtifactError):
    """Serving bundle missing, corrupt, tampered, unsupported or incoherent (exit code 2)."""

class InvalidRequestError(RecError, ValueError):
    """Invalid recommend() input (user_id / top_k / model); exit code 2."""
