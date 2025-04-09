"""
Module implémentant la métrique UniEval pour l'évaluation de texte.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from core.metric_interface import TextMetric

try:
    import torch
    from nltk.tokenize import sent_tokenize
    _UNIEVAL_AVAILABLE = True
    try:
        from unieval.utils.unieval_utils import get_evaluator
        _UNIEVAL_IMPORT_SUCCESS = True
    except ImportError:
        _UNIEVAL_IMPORT_SUCCESS = False
except ImportError:
    _UNIEVAL_AVAILABLE = False
    _UNIEVAL_IMPORT_SUCCESS = False


class UniEvalMetric(TextMetric):
    """
    Implémentation de la métrique UniEval pour l'évaluation de texte.
    
    UniEval est une métrique d'évaluation unifiée qui peut évaluer plusieurs aspects 
    de la qualité du texte (cohérence, cohésion, fidélité, etc.) en utilisant un modèle
    pré-entraîné.
    """
    
    def __init__(self, 
                task: str = "summarization",
                aspects: List[str] = None,
                device: str = None,
                aggregate: str = "mean",
                verbose: bool = False):
        """
        Initialise la métrique UniEval.
        
        Args:
            task: Tâche d'évaluation ('summarization', 'dialogue', 'data2text', 'fact')
            aspects: Liste des aspects à évaluer, spécifiques à la tâche
                Pour summarization: ['coherence', 'consistency', 'fluency', 'relevance']
                Pour dialogue: ['coherence', 'consistency', 'engagingness', 'groundedness']
                Pour data2text: ['correctness', 'coherence', 'fluency']
                Pour fact: ['faithfulness']
            device: Appareil à utiliser ('cuda', 'cpu')
            aggregate: Méthode pour agréger les scores ('mean', 'min', 'max')
            verbose: Afficher des informations détaillées pendant l'évaluation
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _UNIEVAL_AVAILABLE:
            raise ImportError(
                "Les packages 'torch' et 'nltk' sont requis pour utiliser UniEvalMetric. "
                "Installez-les avec 'pip install torch nltk'."
            )
            
        if not _UNIEVAL_IMPORT_SUCCESS:
            raise ImportError(
                "Le package 'unieval' est requis pour utiliser UniEvalMetric. "
                "Installez-le avec 'pip install git+https://github.com/maszhongming/UniEval.git'."
            )
        
        # Vérifier que task est une valeur valide
        valid_tasks = ['summarization', 'dialogue', 'data2text', 'fact']
        if task not in valid_tasks:
            raise ValueError(f"task doit être l'un de {valid_tasks}")
        self.task = task
        
        # Définir les aspects par défaut selon la tâche
        task_aspects = {
            'summarization': ['coherence', 'consistency', 'fluency', 'relevance'],
            'dialogue': ['coherence', 'consistency', 'engagingness', 'groundedness'],
            'data2text': ['correctness', 'coherence', 'fluency'],
            'fact': ['faithfulness']
        }
        
        # Utiliser les aspects spécifiés ou ceux par défaut pour la tâche
        self.aspects = aspects if aspects else task_aspects.get(task, ['coherence'])
        
        # Vérifier que les aspects sont valides pour la tâche choisie
        valid_aspects = task_aspects.get(task, [])
        for aspect in self.aspects:
            if aspect not in valid_aspects:
                raise ValueError(f"Pour la tâche '{task}', les aspects valides sont {valid_aspects}")
        
        # Vérifier que aggregate est une valeur valide
        valid_aggregates = ['mean', 'min', 'max']
        if aggregate not in valid_aggregates:
            raise ValueError(f"aggregate doit être l'un de {valid_aggregates}")
        self.aggregate = aggregate
        
        # Initialiser les évaluateurs pour chaque aspect
        self.evaluators = {}
        for aspect in self.aspects:
            self.evaluators[aspect] = get_evaluator(task=task, aspect=aspect)
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.verbose = verbose
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "unieval"
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Prétraite le texte en le découpant en phrases.
        
        Args:
            text: Texte à prétraiter
            
        Returns:
            Liste des phrases du texte
        """
        return sent_tokenize(text)
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores UniEval entre les références et les candidats.
        
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
        
        # Prétraiter les textes
        ref_sentences = [self._preprocess_text(ref) for ref in references]
        cand_sentences = [self._preprocess_text(cand) for cand in candidates]
        
        # Calculer les scores pour chaque aspect
        aspect_scores = {}
        for aspect in self.aspects:
            evaluator = self.evaluators[aspect]
            
            # Format de données pour l'évaluateur
            samples = []
            for i in range(len(references)):
                sample = {
                    'text': ' '.join(cand_sentences[i]),
                    'reference': ' '.join(ref_sentences[i]),
                    'source': references[i] if self.task in ['summarization', 'fact'] else None
                }
                samples.append(sample)
            
            # Obtenir les scores pour cet aspect
            scores = evaluator.evaluate(samples, device=self.device, verbose=self.verbose)
            aspect_scores[aspect] = scores
        
        # Calculer les scores individuels (agrégeant les scores de tous les aspects)
        individual_scores = []
        for i in range(len(references)):
            aspect_values = [aspect_scores[aspect][i] for aspect in self.aspects]
            
            # Agréger les scores selon la méthode choisie
            if self.aggregate == 'mean':
                score = np.mean(aspect_values)
            elif self.aggregate == 'min':
                score = np.min(aspect_values)
            else:  # self.aggregate == 'max'
                score = np.max(aspect_values)
                
            individual_scores.append(score)
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        # Créer le dictionnaire de résultats
        results = {
            'score': global_score,
            'individual_scores': individual_scores,
            'task': self.task,
            'aggregate': self.aggregate
        }
        
        # Ajouter les scores détaillés par aspect
        for aspect in self.aspects:
            results[aspect] = {
                'score': np.mean(aspect_scores[aspect]),
                'individual_scores': aspect_scores[aspect]
            }
        
        return results
