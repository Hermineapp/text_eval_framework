"""
Module implémentant la métrique BERTScore pour l'évaluation de texte.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

try:
    import torch
    from bert_score import BERTScorer
    _BERT_SCORE_AVAILABLE = True
except ImportError:
    _BERT_SCORE_AVAILABLE = False


class BERTScoreMetric(TextMetric):
    """
    Implémentation de la métrique BERTScore pour l'évaluation de texte.
    
    BERTScore utilise des représentations contextuelles pré-entraînées de BERT
    pour calculer la similarité entre les textes candidats et les références.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_layers: int = None, 
                batch_size: int = 32, all_layers: bool = False, 
                rescale_with_baseline: bool = True, lang: str = "en"):
        """
        Initialise la métrique BERTScore.
        
        Args:
            model_name: Nom du modèle BERT à utiliser
            num_layers: Nombre de couches à utiliser (-1 pour la dernière)
            batch_size: Taille des batchs pour le calcul
            all_layers: Utiliser toutes les couches
            rescale_with_baseline: Rescaler les scores avec une baseline
            lang: Langue des textes
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _BERT_SCORE_AVAILABLE:
            raise ImportError(
                "Les packages 'torch' et 'bert_score' sont requis pour utiliser BERTScoreMetric. "
                "Installez-les avec 'pip install torch bert-score'."
            )
        
        self.model_name = model_name
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.all_layers = all_layers
        self.rescale_with_baseline = rescale_with_baseline
        self.lang = lang
        
        # Initialiser le scorer
        self.scorer = BERTScorer(
            model_type=model_name,
            num_layers=num_layers,
            batch_size=batch_size,
            all_layers=all_layers,
            rescale_with_baseline=rescale_with_baseline,
            lang=lang
        )
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "bert_score"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores BERTScore entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant les scores P, R et F1 globaux et individuels
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Calculer les scores P, R et F1
        P, R, F1 = self.scorer.score(candidates, references)
        
        # Convertir les tenseurs en listes Python
        P_list = P.tolist()
        R_list = R.tolist()
        F1_list = F1.tolist()
        
        return {
            'score': np.mean(F1_list),  # Le score F1 est utilisé comme score global
            'individual_scores': F1_list,
            'precision': {
                'score': np.mean(P_list),
                'individual_scores': P_list
            },
            'recall': {
                'score': np.mean(R_list),
                'individual_scores': R_list
            },
            'f1': {
                'score': np.mean(F1_list),
                'individual_scores': F1_list
            }
        }
