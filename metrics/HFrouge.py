"""
Module implémentant les métriques ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
en utilisant l'implémentation de Hugging Face.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric
from rouge_score import rouge_scorer, scoring

class HuggingFaceRougeMetric(TextMetric):
    """
    Implémentation des métriques ROUGE pour l'évaluation de texte basée sur l'approche Hugging Face.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) est un ensemble de métriques
    qui évaluent la qualité des résumés automatiques en comparant les n-grammes,
    les séquences de mots et les paires de mots avec des références.
    """
    
    def __init__(self, 
                 rouge_types: List[str] = None, 
                 use_stemmer: bool = True,
                 use_aggregator: bool = False):
        """
        Initialise la métrique ROUGE avec l'implémentation Hugging Face.
        
        Args:
            rouge_types: Types de ROUGE à calculer ('rouge1', 'rouge2', 'rougeL', 'rougeLsum')
            use_stemmer: Utiliser un stemmer pour normaliser les mots
            use_aggregator: Utiliser l'agrégateur bootstrap (utile pour les intervalles de confiance)
        """
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.use_stemmer = use_stemmer
        self.use_aggregator = use_aggregator
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=self.rouge_types, 
            use_stemmer=self.use_stemmer
        )
        self.aggregator = scoring.BootstrapAggregator() if use_aggregator else None
    
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
        
        # Stocker les scores individuels par type
        individual_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        # Calculer les scores pour chaque paire référence-candidat
        for ref, cand in zip(references, candidates):
            # Calculer le score
            score = self.scorer.score(ref, cand)
            
            # Ajouter au agrégateur si utilisé
            if self.use_aggregator:
                self.aggregator.add_scores(score)
            
            # Stocker les scores F1 individuels
            for rouge_type in self.rouge_types:
                individual_scores[rouge_type].append(score[rouge_type].fmeasure)
        
        # Résultats finaux
        if self.use_aggregator:
            result = self.aggregator.aggregate()
            # Extraire les scores F1 moyens
            global_scores = {rouge_type: result[rouge_type].mid.fmeasure for rouge_type in self.rouge_types}
        else:
            global_scores = {rouge_type: np.mean(individual_scores[rouge_type]) for rouge_type in self.rouge_types}
        
        # Construire le résultat final au format attendu par l'évaluateur
        final_result = {
            # Score global (utilise rouge1 comme score principal)
            'score': global_scores['rouge1'],
            'individual_scores': individual_scores['rouge1'],
        }
        
        # Ajouter chaque type de ROUGE avec ses scores
        for rouge_type in self.rouge_types:
            key = rouge_type.replace('rouge', '')  # Convertir 'rouge1' en '1'
            final_result[key] = {
                'score': global_scores[rouge_type],
                'individual_scores': individual_scores[rouge_type]
            }
        
        return final_result