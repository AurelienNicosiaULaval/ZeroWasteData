from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class BaseAnalysis(ABC):
    """Abstract base class for all data analyses."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the analysis displayed to the user."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the analysis (e.g., 'Descriptive', 'Correlation')."""
        pass

    @property
    def description(self) -> str:
        """Brief description of what this analysis does."""
        return ""

    @abstractmethod
    def check_applicability(self, df: pd.DataFrame) -> bool:
        """
        Check if the analysis is applicable to the given DataFrame.

        Returns:
            bool: True if applicable, False otherwise.
        """
        pass

    @abstractmethod
    def run(self, df: pd.DataFrame) -> Any:
        """
        Perform the analysis on the data.

        Returns:
            Any: The result of the analysis (can be a dict, dataframe, model, etc.)
        """
        pass

    @abstractmethod
    def render_streamlit(self, df: pd.DataFrame, result: Any) -> Optional[str]:
        """
        Display the results in Streamlit.

        Args:
            df: The original DataFrame.
            result: The result returned by run().

        Returns:
            Optional[str]: Markdown content to be included in the report, or None.
        """
        pass
