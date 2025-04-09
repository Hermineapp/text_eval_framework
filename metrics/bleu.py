"""
Module implémentant la métrique BLEU (Bilingual Evaluation Understudy).
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Any, Union, Optional, Tuple
from core.metric_interface import TextMetric


class BLEUMetric(TextMetric):
    """
    Implémentation de la métrique BLEU pour l'évaluation de texte.
    
    BLEU (Bilingual Evaluation Understudy) est une métrique qui évalue 
    la qualité d'un texte en mesurant la correspondance entre la sortie de modèle
    et une ou plusieurs références, en utilisant des n-grammes.
    """
    
    def __init__(self, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25), 
                smoothing: str = "method1", 
                auto_download: bool = True):
        """
        Initialise la métrique BLEU.
        
        Args:
            weights: Pondérations des n-grammes (1-gramme, 2-grammes, etc.)
            smoothing: Méthode de lissage à utiliser (method0 à method7)
            auto_download: Télécharger automatiquement les ressources nltk nécessaires
        """
        if auto_download:
            nltk.download('punkt', quiet=True)
        
        self.weights = weights
        self.smoothing_function = getattr(SmoothingFunction(), smoothing)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "bleu"
    
    @property
    def requires_tokenization(self) -> bool:
        """Indique si la métrique nécessite une tokenisation spécifique."""
        return True
    
    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenise les textes pour la métrique BLEU.
        
        Args:
            texts: Liste de textes à tokeniser
            
        Returns:
            Liste de textes tokenisés
        """
        result = []
        for text in texts:
            # Vérifier si l'élément est déjà une liste ou un texte
            if isinstance(text, list):
                # Si c'est déjà une liste, vérifier si ce sont des tokens ou une liste de références
                if text and isinstance(text[0], str):
                    # C'est déjà une liste de tokens
                    result.append(text)
                else:
                    # C'est une liste de références, prendre la première
                    result.append(text[0] if text else [])
            else:
                # C'est une chaîne de caractères, la tokeniser
                result.append(nltk.word_tokenize(text.lower()))
        return result
    
    def compute(self, references: List[List[str]], candidates: List[List[str]], 
               already_tokenized: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Calcule le score BLEU entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence (tokenisés ou non)
            candidates: Liste de textes candidats (tokenisés ou non)
            already_tokenized: Indique si les textes sont déjà tokenisés
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if not already_tokenized and not self.requires_tokenization:
            tokenized_refs = references
            tokenized_cands = candidates
        elif not already_tokenized:
            tokenized_refs = self.preprocess(references)
            tokenized_cands = self.preprocess(candidates)
        else:
            tokenized_refs = references
            tokenized_cands = candidates
        
        scores = []
        for cand, ref in zip(tokenized_cands, tokenized_refs):
            # Pour BLEU, les références doivent être une liste de listes
            if isinstance(ref[0], str):  # si ref est déjà tokenisé
                ref_list = [ref]
            else:  # si ref est une liste de références tokenisées
                ref_list = ref
                
            score = sentence_bleu(ref_list, cand, 
                                 weights=self.weights,
                                 smoothing_function=self.smoothing_function)
            scores.append(score)
            
        return {
            'score': sum(scores) / len(scores) if scores else 0,
            'individual_scores': scores
        }
