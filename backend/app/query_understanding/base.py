"""Abstract base interface for query understanding and semantic enrichment."""

from abc import ABC, abstractmethod
from backend.app.models.search import QueryUnderstandingResult


class BaseQueryProcessor(ABC):
    """Abstract interface for analyzing, normalizing, and expanding search queries."""

    @abstractmethod
    def process(self, raw_query: str) -> QueryUnderstandingResult:
        """Process a raw user query string into structured search intent."""
        pass
