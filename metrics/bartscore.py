"""
Module implémentant la métrique BartScore pour l'évaluation de texte.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.metric_interface import TextMetric

try:
    import torch
    from transformers import BartTokenizer, BartForConditionalGeneration
    _BARTSCORE_AVAILABLE = True
except ImportError:
    _BARTSCORE_AVAILABLE = False


class BartScoreMetric(TextMetric):
    """
    Implémentation de la métrique BartScore pour l'évaluation de texte.
    
    BartScore utilise un modèle BART pour évaluer la qualité d'un texte généré
    en calculant la probabilité conditionnelle du texte candidat étant donné le texte de référence.
    """
    
    def __init__(self, 
                model_name: str = "facebook/bart-large-cnn", 
                device: str = None,
                max_length: int = 1024,
                batch_size: int = 4,
                direction: str = "avg"):
        """
        Initialise la métrique BartScore.
        
        Args:
            model_name: Nom du modèle BART à utiliser
            device: Appareil à utiliser ('cuda', 'cpu')
            max_length: Longueur maximale des textes en tokens
            batch_size: Taille du batch pour les calculs
            direction: Direction d'évaluation ('src2tgt', 'tgt2src', ou 'avg')
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _BARTSCORE_AVAILABLE:
            raise ImportError(
                "Les packages 'torch' et 'transformers' sont requis pour utiliser BartScoreMetric. "
                "Installez-les avec 'pip install torch transformers'."
            )
            
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Vérifier que direction est une valeur valide
        assert direction in ['src2tgt', 'tgt2src', 'avg'], "direction doit être 'src2tgt', 'tgt2src' ou 'avg'"
        self.direction = direction
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Charger le modèle et le tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "bartscore"
    
    def _compute_bartscore(self, srcs: List[str], tgts: List[str]) -> List[float]:
        """
        Calcule le score BART d'une liste de références vers une liste de candidats.
        
        Args:
            srcs: Liste de textes source (références)
            tgts: Liste de textes cible (candidats)
            
        Returns:
            Liste des scores (log-probabilités) pour chaque paire (src, tgt)
        """
        scores = []
        
        for i in range(0, len(srcs), self.batch_size):
            src_batch = srcs[i:i+self.batch_size]
            tgt_batch = tgts[i:i+self.batch_size]
            
            # Tokenize les textes
            with torch.no_grad():
                inputs = self.tokenizer(src_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Tokenize les textes cibles
                targets = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Calculer les log-probabilités
                outputs = self.model(**inputs, labels=targets["input_ids"])
                log_likelihood = -outputs.loss.item() * targets["input_ids"].shape[0] * targets["input_ids"].shape[1]
                
                # Normaliser par la longueur
                for j in range(len(tgt_batch)):
                    tgt_len = targets["attention_mask"][j].sum().item()
                    scores.append(log_likelihood / tgt_len)
        
        return scores
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores BartScore entre les références et les candidats.
        
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
        
        # Calculer les scores dans les deux directions
        if self.direction == 'src2tgt' or self.direction == 'avg':
            ref2cand_scores = self._compute_bartscore(references, candidates)
        else:
            ref2cand_scores = []
            
        if self.direction == 'tgt2src' or self.direction == 'avg':
            cand2ref_scores = self._compute_bartscore(candidates, references)
        else:
            cand2ref_scores = []
        
        # Calculer les scores individuels selon la direction choisie
        if self.direction == 'src2tgt':
            individual_scores = ref2cand_scores
        elif self.direction == 'tgt2src':
            individual_scores = cand2ref_scores
        else:  # self.direction == 'avg'
            individual_scores = [(s1 + s2) / 2 for s1, s2 in zip(ref2cand_scores, cand2ref_scores)]
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'direction': self.direction,
            'model': self.model_name
        }
