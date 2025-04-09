"""
Interface de base pour toutes les métriques d'évaluation de texte.
Ce module définit l'API que toutes les métriques doivent implémenter.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class TextMetric(ABC):
    """Interface de base pour toutes les métriques d'évaluation de texte."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la métrique."""
        pass
    
    @abstractmethod
    def compute(self, references: List[Any], candidates: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Calcule la métrique entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats à évaluer
            **kwargs: Paramètres supplémentaires spécifiques à la métrique
            
        Returns:
            dict: Résultats avec au moins les clés 'score' (global) et 
                 'individual_scores' (par instance)
        """
        pass
    
    @property
    def requires_tokenization(self) -> bool:
        """Indique si la métrique nécessite une tokenisation spécifique."""
        return False
    
    def preprocess(self, texts: List[str]) -> List[Any]:
        """
        Prétraitement optionnel des textes.
        
        Args:
            texts: Liste de textes à prétraiter
            
        Returns:
            Liste de textes prétraités (potentiellement tokenisés)
        """
        return texts
