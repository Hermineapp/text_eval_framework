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
        """
        Calcule les scores ROUGE entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant les scores globaux et individuels pour chaque type de ROUGE
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        all_scores = []
        for ref, cand in zip(references, candidates):
            all_scores.append(self.scorer.score(ref, cand))
        
        # Organiser les résultats par type de ROUGE
        results = {}
        individual_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for score in all_scores:
            for rouge_type in self.rouge_types:
                # Pour chaque type, extraire les scores F1, de précision et de rappel
                individual_scores[rouge_type].append(score[rouge_type].fmeasure)
        
        # Calculer les scores moyens
        for rouge_type in self.rouge_types:
            scores = individual_scores[rouge_type]
            results[rouge_type] = {
                'score': np.mean(scores),
                'individual_scores': scores
            }
        
        # Créer un score global (moyenne des F1 de tous les types de ROUGE)
        all_f1 = [score for type_scores in individual_scores.values() for score in type_scores]
        
        return {
            'score': np.mean(all_f1),
            'individual_scores': [np.mean([s[t].fmeasure for t in self.rouge_types]) for s in all_scores],
            'types': results
        }
