"""
Module implémentant la métrique SentenceBERT pour la similarité sémantique.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


class SentenceBERTMetric(TextMetric):
    """
    Implémentation de la métrique SentenceBERT pour l'évaluation de texte.
    
    SentenceBERT génère des embeddings de phrases et calcule la similarité
    cosinus entre les phrases de référence et les phrases candidates.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32,
                similarity_metric: str = 'cosine', normalize_embeddings: bool = True,
                device: Optional[str] = None):
        """
        Initialise la métrique SentenceBERT.
        
        Args:
            model_name: Nom du modèle SentenceBERT à utiliser
            batch_size: Taille des batchs pour le calcul des embeddings
            similarity_metric: Métrique de similarité à utiliser ('cosine', 'euclidean', 'dot_product')
            normalize_embeddings: Normaliser les embeddings avant de calculer la similarité
            device: Appareil à utiliser pour les calculs ('cpu', 'cuda', etc.)
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "Les packages 'sentence-transformers' et 'torch' sont requis pour utiliser SentenceBERTMetric. "
                "Installez-les avec 'pip install sentence-transformers torch'."
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.similarity_metric = similarity_metric
        self.normalize_embeddings = normalize_embeddings
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Charger le modèle
        self.model = SentenceTransformer(model_name, device=self.device)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return f"sbert_{self.model_name.replace('-', '_')}"
    
    def _compute_similarity(self, 
                          ref_embeddings: torch.Tensor, 
                          cand_embeddings: torch.Tensor) -> List[float]:
        """
        Calcule la similarité entre les embeddings de référence et les embeddings candidats.
        
        Args:
            ref_embeddings: Embeddings des références
            cand_embeddings: Embeddings des candidats
            
        Returns:
            Liste des scores de similarité
        """
        if self.similarity_metric == 'cosine':
            # Similarité cosinus
            if self.normalize_embeddings:
                scores = util.cos_sim(ref_embeddings, cand_embeddings)
            else:
                normalized_ref = torch.nn.functional.normalize(ref_embeddings, p=2, dim=1)
                normalized_cand = torch.nn.functional.normalize(cand_embeddings, p=2, dim=1)
                scores = util.cos_sim(normalized_ref, normalized_cand)
                
        elif self.similarity_metric == 'euclidean':
            # Distance euclidienne (convertie en similarité)
            distances = util.pairwise_distance(ref_embeddings, cand_embeddings)
            # Convertir les distances en similarités (plus la distance est petite, plus la similarité est grande)
            scores = 1.0 / (1.0 + distances)
            
        elif self.similarity_metric == 'dot_product':
            # Produit scalaire
            scores = torch.sum(ref_embeddings * cand_embeddings, dim=1)
            
        else:
            raise ValueError(f"Métrique de similarité non reconnue: {self.similarity_metric}")
        
        # Convertir en liste Python
        if scores.dim() == 2:
            # Si scores est une matrice, prendre la diagonale
            scores = torch.diag(scores)
            
        return scores.cpu().tolist()
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule la similarité SentenceBERT entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Calculer les embeddings
        ref_embeddings = self.model.encode(
            references, 
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        cand_embeddings = self.model.encode(
            candidates,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Calculer les scores de similarité
        individual_scores = self._compute_similarity(ref_embeddings, cand_embeddings)
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'model': self.model_name,
            'similarity_metric': self.similarity_metric
        }
