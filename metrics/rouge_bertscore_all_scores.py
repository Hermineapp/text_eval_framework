"""
Module implémentant des versions complètes des métriques ROUGE et BERTScore.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

# ROUGE implementation
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

# BERTScore implementation    
try:
    import torch
    from bert_score import BERTScorer
    _BERT_SCORE_AVAILABLE = True
except ImportError:
    _BERT_SCORE_AVAILABLE = False


class EnhancedROUGEMetric(TextMetric):
    """
    Implémentation complète des métriques ROUGE pour l'évaluation de texte.
    
    Cette version calcule ROUGE-1, ROUGE-2, et ROUGE-L avec précision, rappel et F1.
    """
    
    def __init__(self, 
                rouge_types: List[str] = None, 
                use_stemmer: bool = True,
                split_summaries: bool = True):
        """
        Initialise la métrique ROUGE améliorée.
        
        Args:
            rouge_types: Types de ROUGE à calculer ('rouge1', 'rouge2', 'rougeL')
            use_stemmer: Utiliser un stemmer pour normaliser les mots
            split_summaries: Diviser les résumés en phrases pour ROUGE-L
            
        Raises:
            ImportError: Si le package rouge_score n'est pas installé
        """
        if not _ROUGE_AVAILABLE:
            raise ImportError(
                "Le package 'rouge_score' est requis pour utiliser EnhancedROUGEMetric. "
                "Installez-le avec 'pip install rouge_score'."
            )
        
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.use_stemmer = use_stemmer
        self.split_summaries = split_summaries
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types, 
            use_stemmer=use_stemmer,
            split_summaries=split_summaries
        )
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "enhanced_rouge"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores ROUGE complets entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant les scores globaux et individuels pour chaque type de ROUGE et chaque mesure (P/R/F1)
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        all_scores = []
        for ref, cand in zip(references, candidates):
            all_scores.append(self.scorer.score(ref, cand))
        
        # Organisation des résultats par type de ROUGE
        results = {}
        for rouge_type in self.rouge_types:
            # Stocker les scores de précision, rappel et F1
            precision_scores = [score[rouge_type].precision for score in all_scores]
            recall_scores = [score[rouge_type].recall for score in all_scores]
            f1_scores = [score[rouge_type].fmeasure for score in all_scores]
            
            results[rouge_type] = {
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
        
        # Calculer le score global (moyenne des scores F1)
        avg_f1_scores = [np.mean([score[t].fmeasure for t in self.rouge_types]) for score in all_scores]
        
        # Créer un dictionnaire pour les scores F1 individuels pour chaque type
        individual_type_scores = {}
        for rouge_type in self.rouge_types:
            individual_type_scores[f'{rouge_type}_f1'] = [score[rouge_type].fmeasure for score in all_scores]
            individual_type_scores[f'{rouge_type}_precision'] = [score[rouge_type].precision for score in all_scores]
            individual_type_scores[f'{rouge_type}_recall'] = [score[rouge_type].recall for score in all_scores]
        
        # Résultat global
        return {
            'score': np.mean(avg_f1_scores),
            'individual_scores': avg_f1_scores,
            'individual_type_scores': individual_type_scores,
            'types': results
        }


class EnhancedBERTScoreMetric(TextMetric):
    """
    Implémentation complète de la métrique BERTScore pour l'évaluation de texte.
    
    Cette version renvoie les scores de précision, rappel et F1.
    """
    
    def __init__(self, 
                model_name: str = "bert-base-uncased", 
                num_layers: int = None, 
                batch_size: int = 32,
                lang: str = "en",
                rescale_with_baseline: bool = True,
                device: str = None):
        """
        Initialise la métrique BERTScore améliorée.
        
        Args:
            model_name: Nom du modèle BERT à utiliser
            num_layers: Nombre de couches à utiliser (-1 pour la dernière)
            batch_size: Taille des batchs pour le calcul
            lang: Langue des textes
            rescale_with_baseline: Rescaler les scores avec une baseline
            device: Appareil à utiliser ('cuda', 'cpu')
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _BERT_SCORE_AVAILABLE:
            raise ImportError(
                "Les packages 'torch' et 'bert_score' sont requis pour utiliser EnhancedBERTScoreMetric. "
                "Installez-les avec 'pip install torch bert-score'."
            )
        
        self.model_name = model_name
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialiser le scorer
        self.scorer = BERTScorer(
            model_type=model_name,
            num_layers=num_layers,
            batch_size=batch_size,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            device=self.device
        )
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "enhanced_bert_score"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores BERTScore complets entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant les scores globaux et individuels pour P/R/F1
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
        
        # Créer le dictionnaire de résultats
        results = {
            'score': np.mean(F1_list),  # Le score F1 est utilisé comme score global par défaut
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
        
        return results
