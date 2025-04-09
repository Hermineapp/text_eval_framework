"""
Module principal du framework d'évaluation de texte.
Contient les classes fondamentales pour la configuration et l'exécution des évaluations.
"""

from .metric_interface import TextMetric
from .evaluator import TextEvaluator
from .correlation import MetricCorrelation
from .report import Report

__all__ = ['TextMetric', 'TextEvaluator', 'MetricCorrelation', 'Report']
