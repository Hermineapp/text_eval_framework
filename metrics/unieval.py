"""
Module implémentant la métrique UniEval pour l'évaluation de texte.
"""

import os
import sys
import logging
from typing import List, Dict, Any
import numpy as np
import torch
from core.metric_interface import TextMetric

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniEvalMetric(TextMetric):
    """
    Implémentation de la métrique UniEval pour l'évaluation de texte.
    
    UniEval est un framework d'évaluation unifié qui peut évaluer 
    différentes tâches de génération de texte selon de multiples dimensions.
    """
    
    def __init__(self, task: str = "summarization", 
                aspects: List[str] = None,
                verbose: bool = False,
                device: str = None,
                cache_dir: str = None):
        """
        Initialise la métrique UniEval.
        
        Args:
            task: Tâche d'évaluation ("summarization", "dialogue", "data2text", "fact")
            aspects: Liste des aspects à évaluer (cohérence, fidélité, etc.)
            verbose: Afficher les détails pendant l'évaluation
            device: Appareil à utiliser ('cuda', 'cpu')
            cache_dir: Répertoire de cache pour les modèles
        """
        # Définir le chemin vers les modules UniEval
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.unieval_dir = os.path.join(current_dir, 'unieval_lib')
        
        # Ajouter temporairement au PYTHONPATH
        if self.unieval_dir not in sys.path:
            sys.path.insert(0, self.unieval_dir)
        
        try:
            # Importer les modules UniEval
            from unieval_utils import convert_to_json
            from metric.evaluator import get_evaluator
            
            self._convert_to_json = convert_to_json
            self._get_evaluator = get_evaluator
            self._unieval_available = True
        except Exception as e:
            logger.error(f"Erreur d'importation de UniEval: {e}")
            self._unieval_available = False
            raise ImportError(f"UniEval n'est pas disponible. Erreur: {e}")
        
        self.task = task
        self.verbose = verbose
        # check if cuda is available
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        #self.device = device if device else 'cuda:0'

        self.cache_dir = cache_dir
        
        # Aspects à évaluer selon la tâche
        self.aspects = aspects or self._get_default_aspects(task)
        
        # Initialiser l'évaluateur UniEval
        try:
            self.evaluator = self._get_evaluator(
                task=task, 
                max_length=1024,
                device=self.device,
                cache_dir=self.cache_dir
            )
            logger.info(f"UniEval initialisé pour la tâche '{task}' avec les aspects {self.aspects}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de UniEval: {e}")
            raise
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "unieval"
    
    def _get_default_aspects(self, task: str) -> List[str]:
        """
        Détermine les aspects par défaut à évaluer en fonction de la tâche.
        
        Args:
            task: Tâche d'évaluation
            
        Returns:
            Liste des aspects par défaut
        """
        if task == "summarization":
            return ["coherence", "consistency", "fluency", "relevance"]
        elif task == "dialogue":
            return ["naturalness", "coherence", "engagingness", "groundedness", "understandability"]
        elif task == "data2text":
            return ["naturalness", "informativeness"]
        elif task == "fact":
            return ["consistency"]
        else:
            return ["coherence", "fluency"]  # Aspects de base pour toute tâche
    
    def compute(self, references: List[str], candidates: List[str], sources: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores UniEval entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            sources: Liste de textes source (requis pour certaines tâches)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant les scores globaux et individuels pour chaque aspect
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Vérification et préparation des sources selon la tâche
        if sources is None and self.task in ["summarization", "fact"]:
            sources = references  # Utiliser les références comme sources par défaut
        
        # Préparer les données pour UniEval selon la tâche
        try:
            if self.task == "summarization" or self.task == "fact":
                data = self._convert_to_json(
                    output_list=candidates,
                    src_list=sources,
                    ref_list=references
                )
            elif self.task == "dialogue":
                # Pour le dialogue, on a besoin de l'historique et du contexte
                # Ici on utilise les références comme historique et les sources comme contexte
                data = self._convert_to_json(
                    output_list=candidates,
                    src_list=references,  # historique du dialogue
                    context_list=sources if sources else ['' for _ in range(len(candidates))]
                )
            elif self.task == "data2text":
                data = self._convert_to_json(
                    output_list=candidates,
                    ref_list=references
                )
            else:
                data = self._convert_to_json(
                    output_list=candidates,
                    src_list=sources if sources else references,
                    ref_list=references
                )
            
            # Évaluer avec UniEval
            # Note: l'argument 'aspects' n'est pas supporté par la méthode evaluate()
            # Nous allons utiliser seulement 'dims' qui est le paramètre supporté
            dims = self.aspects
            
            # Évaluer avec les dimensions spécifiées
            if self.task == "fact":
                # Pour fact, pas besoin de spécifier les dimensions
                eval_scores = self.evaluator.evaluate(data, print_result=self.verbose)
            else:
                # Pour les autres tâches, spécifier les dimensions avec dims
                eval_scores = self.evaluator.evaluate(
                    data, 
                    dims=dims,
                    print_result=self.verbose
                )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation avec UniEval: {e}")
            raise
        
        # Organiser les résultats
        results = {}
        aspect_scores = {}
        individual_scores = []
        
        # Calculer un score global moyen pour tous les exemples
        global_score = 0.0
        aspect_count = 0
        
        # Parcourir les scores et extraire les valeurs
        for i in range(len(candidates)):
            example_score = 0.0
            example_aspect_count = 0
            
            if i < len(eval_scores):
                for aspect in dims:
                    if aspect in eval_scores[i]:
                        score = float(eval_scores[i][aspect])
                        
                        # Initialiser la liste pour cet aspect s'il n'existe pas
                        if aspect not in aspect_scores:
                            aspect_scores[aspect] = {
                                'score': 0.0,
                                'individual_scores': []
                            }
                        
                        # Ajouter le score individuel
                        aspect_scores[aspect]['individual_scores'].append(score)
                        
                        # Mise à jour du score moyen pour l'exemple
                        example_score += score
                        example_aspect_count += 1
            
            # Calculer le score moyen pour cet exemple
            if example_aspect_count > 0:
                avg_example_score = example_score / example_aspect_count
                individual_scores.append(avg_example_score)
        
        # Calculer les scores moyens par aspect
        for aspect, data in aspect_scores.items():
            if data['individual_scores']:
                data['score'] = float(np.mean(data['individual_scores']))
                global_score += data['score']
                aspect_count += 1
        
        # Calculer le score global final
        if aspect_count > 0:
            global_score = global_score / aspect_count
        
        # Créer le dictionnaire de résultats final
        results = {
            'score': global_score,
            'individual_scores': individual_scores,
            'aspects': aspect_scores,
            'task': self.task
        }
        
        return results