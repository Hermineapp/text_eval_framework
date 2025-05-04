"""
Module implémentant les métriques ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


class ROUGEMetric(TextMetric):
    """
    Implémentation des métriques ROUGE pour l'évaluation de texte.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) est un ensemble de métriques
    qui évaluent la qualité des résumés automatiques en comparant les n-grammes,
    les séquences de mots et les paires de mots avec des références.
    """
    
    def __init__(self, rouge_types: List[str] = None, use_stemmer: bool = True):
        """
        Initialise la métrique ROUGE.
        
        Args:
            rouge_types: Types de ROUGE à calculer ('rouge1', 'rouge2', 'rougeL')
            use_stemmer: Utiliser un stemmer pour normaliser les mots
            
        Raises:
            ImportError: Si le package rouge_score n'est pas installé
        """
        if not _ROUGE_AVAILABLE:
            raise ImportError(
                "Le package 'rouge_score' est requis pour utiliser ROUGEMetric. "
                "Installez-le avec 'pip install rouge_score'."
            )
        
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "rouge"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )

        # Calculate scores for each pair
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, cand in zip(references, candidates):
            score = self.scorer.score(ref, cand)
            rouge1_scores.append(score['rouge1'].fmeasure)
            rouge2_scores.append(score['rouge2'].fmeasure)
            rougeL_scores.append(score['rougeL'].fmeasure)
        
        # Return a simple, clear structure
        return {
            'score': np.mean(rouge1_scores),  # Default score is ROUGE-1 F1
            'individual_scores': rouge1_scores,
            'rouge1': {
                'score': np.mean(rouge1_scores),
                'individual_scores': rouge1_scores
            },
            'rouge2': {
                'score': np.mean(rouge2_scores),
                'individual_scores': rouge2_scores
            },
            'rougeL': {
                'score': np.mean(rougeL_scores),
                'individual_scores': rougeL_scores
            }
        }