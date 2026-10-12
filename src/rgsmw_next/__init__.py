"""LPWS-RDF implementation with the historical rgsmw package namespace."""

from .distance import FittedMetric
from .learning import GlobalSimplexLearner, PersonalizedSimplexLearner
from .types import EntityRecord, PredicateSpec, Preference

__all__ = [
    "EntityRecord",
    "FittedMetric",
    "GlobalSimplexLearner",
    "PersonalizedSimplexLearner",
    "PredicateSpec",
    "Preference",
]

__version__ = "0.1.0"
