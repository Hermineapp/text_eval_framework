"""
Module implémentant la métrique METEOR (Metric for Evaluation of Translation with Explicit ORdering).
"""

import nltk
from nltk.translate.meteor_score import meteor_score
from typing import List, Dict, Any, Optional
import numpy as np
from core.metric_interface import TextMetric


class METEORMetric(TextMetric):
    """
    Implémentation de la métrique METEOR pour l'évaluation de texte.
    
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) est une métrique
    qui évalue la qualité des traductions en tenant compte de la correspondance exacte des mots,
    des stems, des synonymes et de la paraphrase.
    """
    
    def __init__(self, language: str = 'en', alpha: float = 0.9, beta: float = 3.0, 
                gamma: float = 0.5, use_synonyms: bool = True, auto_download: bool = True):
        """
        Initialise la métrique METEOR.
        
        Args:
            language: Langue des textes ('en' pour l'anglais)
            alpha: Paramètre de pénalité pour la fragmentation
            beta: Paramètre de pénalité pour la fragmentation
            gamma: Paramètre de pénalité pour la fragmentation
            use_synonyms: Utiliser WordNet pour les synonymes (si disponible)
            auto_download: Télécharger automatiquement les ressources NLTK nécessaires
        """
        self.language = language
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_synonyms = use_synonyms
        
        if auto_download:
            # Télécharger les ressources NLTK nécessaires
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "meteor"
    
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
    
    def compute(self, references: List[Any], candidates: List[Any], 
               already_tokenized: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Calcule le score METEOR entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence (tokenisés ou non)
            candidates: Liste de textes candidats (tokenisés ou non)
            already_tokenized: Indique si les textes sont déjà tokenisés
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if not already_tokenized and self.requires_tokenization:
            tokenized_refs = self.preprocess(references)
            tokenized_cands = self.preprocess(candidates)
        else:
            tokenized_refs = references
            tokenized_cands = candidates
        
        individual_scores = []
        
        for i, (cand, ref) in enumerate(zip(tokenized_cands, tokenized_refs)):
            # Pour METEOR, il faut s'assurer que la référence est une liste de listes de tokens
            if isinstance(ref[0], str):  # Si ref est une liste de tokens
                ref_list = [ref]
            else:  # Si ref est déjà une liste de listes de tokens
                ref_list = ref
                
            try:
                # Calculer le score METEOR
                score = meteor_score(
                    ref_list, 
                    cand,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma
                )
                individual_scores.append(score)
            except Exception as e:
                print(f"Erreur lors du calcul de METEOR pour l'instance {i}: {e}")
                individual_scores.append(0.0)
                
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores) if individual_scores else 0.0
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'params': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'language': self.language
            }
        }
