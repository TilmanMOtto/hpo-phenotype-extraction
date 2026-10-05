"""Abstract base class for evaluation datasets."""

from abc import ABC, abstractmethod


class EvalDataset(ABC):
    """
    Base class for dataset-specific evaluation loaders.

    Subclasses implement load_ground_truth() to return the dataset's
    ground truth in a standardised format, and evaluate() to run
    the full evaluation pipeline including metrics and optional saving.
    """

    @abstractmethod
    def load_ground_truth(self) -> dict:
        """
        Load and return ground truth for this dataset.

        Returns dataset-specific ground truth (e.g., dict[id → list[hpo_codes]]).
        """
        ...

    @abstractmethod
    def get_predictions(self, response_dict: dict, target_symptoms: list[str]) -> dict:
        """
        Convert raw response_dict to per-sample predictions.

        Returns dataset-specific predictions matching the ground-truth format.
        """
        ...

    @abstractmethod
    def evaluate(self, response_dict: dict, target_symptoms: list[str], output_dir: str | None = None) -> dict:
        """
        Run full evaluation: load GT, extract predictions, compute metrics, optionally save.

        Returns metrics dict.
        """
        ...
