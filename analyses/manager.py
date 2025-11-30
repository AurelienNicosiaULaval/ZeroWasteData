from typing import List, Type
import pandas as pd
from .base import BaseAnalysis


class AnalysisManager:
    def __init__(self):
        self.analyses: List[Type[BaseAnalysis]] = []

    def register_analysis(self, analysis_cls: Type[BaseAnalysis]):
        """Register a new analysis class."""
        self.analyses.append(analysis_cls)

    def get_applicable_analyses(self, df: pd.DataFrame) -> List[BaseAnalysis]:
        """Return a list of instantiated analyses applicable to the dataframe."""
        applicable = []
        for cls in self.analyses:
            instance = cls()
            if instance.check_applicability(df):
                applicable.append(instance)
        return applicable

    def get_analysis_by_name(self, name: str) -> Type[BaseAnalysis]:
        """Get an analysis class by its name."""
        for cls in self.analyses:
            if cls().name == name:
                return cls
        return None
