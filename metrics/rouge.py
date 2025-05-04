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
            y compris les scores F1, précision et rappel
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

        # Pour chaque métrique, nous allons créer des listes pour F1, précision et rappel
        individual_f1_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        individual_precision_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        individual_recall_scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for score in all_scores:
            for rouge_type in self.rouge_types:
                # Extraire les scores F1, de précision et de rappel
                individual_f1_scores[rouge_type].append(score[rouge_type].fmeasure)
                individual_precision_scores[rouge_type].append(score[rouge_type].precision)
                individual_recall_scores[rouge_type].append(score[rouge_type].recall)

        # Calculer les scores moyens pour chaque type de ROUGE
        for rouge_type in self.rouge_types:
            f1_scores = individual_f1_scores[rouge_type]
            precision_scores = individual_precision_scores[rouge_type]
            recall_scores = individual_recall_scores[rouge_type]

            results[rouge_type] = {
                'score': np.mean(f1_scores),  # Le score principal reste F1
                'individual_scores': f1_scores,
                'precision': {
                    'score': np.mean(precision_scores),
                    'individual_scores': precision_scores
                },
                'recall': {
                    'score': np.mean(recall_scores),
                    'individual_scores': recall_scores
                },
                'f1': {
                    'score': np.mean(f1_scores),
                    'individual_scores': f1_scores
                }
            }

        # Calculer des scores combinés pour une valeur f1 globale
        primary_type = self.rouge_types[0]  # Utiliser le premier type comme principal (généralement rouge1)
        rouge2 = self.rouge_types[1] if len(self.rouge_types) > 1 else None
        rougeL = self.rouge_types[2] if len(self.rouge_types) > 2 else None
        # Retourner les résultats complets avec structure appropriée pour l'évaluateur
        return {
            'score': results[primary_type]['f1']['score'],  # Le score principal est F1 du premier type
            'individual_scores': results[primary_type]['f1']['individual_scores'],
            'precision': {  # Structure attendue par l'évaluateur
                'score': results[primary_type]['precision']['score'],
                'individual_scores': results[primary_type]['precision']['individual_scores']
            },
            'recall': {  # Structure attendue par l'évaluateur
                'score': results[rouge2]['f1']['score'],
                'individual_scores': results[rouge2]['f1']['individual_scores']
            },
            'f1': {  # Structure attendue par l'évaluateur
                'score': results[rougeL]['f1']['score'],
                'individual_scores': results[rougeL]['f1']['individual_scores']
            },
            'types': results  # Tous les détails par type
        }