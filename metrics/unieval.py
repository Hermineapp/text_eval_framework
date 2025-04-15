"""
Module implémentant la métrique UniEval pour l'évaluation de texte.
"""

import os
import sys
import logging
import importlib.util
from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

# Définir le chemin vers les modules UniEval
current_dir = os.path.dirname(os.path.abspath(__file__))
unieval_dir = os.path.join(current_dir, 'unieval_lib')
metric_dir = os.path.join(unieval_dir, 'metric')

# Ajouter les deux répertoires au path de Python
if unieval_dir not in sys.path:
    sys.path.insert(0, unieval_dir)
if metric_dir not in sys.path:
    sys.path.insert(0, metric_dir)

# Importer les modules dynamiquement avec un chemin complet
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Ajouter au cache des modules
    spec.loader.exec_module(module)
    return module

# Essayer d'importer les fonctions nécessaires
try:
    # Importer utils
    utils_path = os.path.join(unieval_dir, 'unieval_utils.py')
    utils_module = import_from_file('unieval_utils', utils_path)
    convert_to_json = utils_module.convert_to_json
    
    # Importer evaluator
    evaluator_path = os.path.join(metric_dir, 'evaluator.py')
    evaluator_module = import_from_file('evaluator', evaluator_path)
    get_evaluator = evaluator_module.get_evaluator
    
    _UNIEVAL_AVAILABLE = True
    logging.info("UniEval importé avec succès via importation dynamique")
except Exception as e:
    logging.warning(f"Erreur d'importation de UniEval: {e}")
    _UNIEVAL_AVAILABLE = False
    
# Le reste de votre classe reste inchangé
class UniEvalMetric(TextMetric):
    """
    Implémentation de la métrique UniEval pour l'évaluation de texte.
    
    UniEval est un framework d'évaluation unifié qui peut évaluer 
    différentes tâches de génération de texte selon de multiples dimensions.
    """
    
    def __init__(self, task: str = "summarization", 
                aspects: List[str] = None,
                verbose: bool = False,
                device: str = None):
        """
        Initialise la métrique UniEval.
        
        Args:
            task: Tâche d'évaluation ("summarization", "dialogue", "data2text", etc.)
            aspects: Liste des aspects à évaluer (cohérence, fidélité, etc.)
            verbose: Afficher les détails pendant l'évaluation
            device: Appareil à utiliser ('cuda', 'cpu')
            
        Raises:
            ImportError: Si UniEval n'est pas disponible
        """
        if not _UNIEVAL_AVAILABLE:
            # Vérifier les fichiers et répertoires qui pourraient manquer
            unieval_files = []
            if os.path.exists(unieval_dir):
                unieval_files = os.listdir(unieval_dir)
            
            metric_dir = os.path.join(unieval_dir, 'metric')
            metric_files = []
            if os.path.exists(metric_dir):
                metric_files = os.listdir(metric_dir)
            
            raise ImportError(
                f"UniEval n'est pas disponible. Assurez-vous que les fichiers de la bibliothèque "
                f"sont présents dans le dossier 'metrics/unieval_lib'.\n"
                f"Fichiers trouvés dans unieval_lib: {unieval_files}\n"
                f"Fichiers trouvés dans unieval_lib/metric: {metric_files}"
            )
        
        self.task = task
        self.verbose = verbose
        self.device = device
        
        # Aspects à évaluer selon la tâche
        self.aspects = aspects or self._get_default_aspects(task)
        
        # Initialiser l'évaluateur UniEval
        try:
            self.evaluator = get_evaluator(task, device=device)
            logging.info(f"UniEval initialisé pour la tâche '{task}' avec les aspects {self.aspects}")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de UniEval: {e}")
            logging.error(f"Détails de l'erreur: {str(e)}")
            logging.error(f"Vérifiez que tous les fichiers nécessaires sont présents dans 'metrics/unieval_lib'")
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
            return ["coherence", "consistency", "fluency", "engagingness"]
        elif task == "data2text":
            return ["coherence", "consistency", "fluency", "relevance"]
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
        
        # Pour certaines tâches comme le résumé, les sources sont nécessaires
        if self.task in ["summarization", "data2text"] and sources is None:
            raise ValueError(f"Des textes source sont requis pour la tâche '{self.task}'")
        
        # Si aucune source n'est fournie, utiliser les références comme sources
        src_list = sources if sources is not None else references
        
        # Préparer les données pour UniEval
        try:
            data = convert_to_json(
                output_list=candidates,
                ref_list=references,
                src_list=src_list
            )
            
            # Évaluer avec UniEval
            eval_results = self.evaluator.evaluate(
                data,
                print_result=self.verbose,
                aspects=self.aspects
            )
            
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation avec UniEval: {e}")
            raise
        
        # Organiser les résultats
        results = {}
        aspect_scores = {}
        individual_scores = []
        
        # Extraire les scores pour chaque aspect
        for aspect in self.aspects:
            if aspect in eval_results:
                aspect_scores[aspect] = {
                    'score': float(np.mean(eval_results[aspect])),
                    'individual_scores': [float(score) for score in eval_results[aspect]]
                }
        
        # Calculer un score global (moyenne des aspects)
        global_scores = []
        for i in range(len(candidates)):
            instance_scores = []
            for aspect in self.aspects:
                if aspect in eval_results and i < len(eval_results[aspect]):
                    instance_scores.append(eval_results[aspect][i])
            if instance_scores:
                score = float(np.mean(instance_scores))
                individual_scores.append(score)
                global_scores.append(score)
        
        # Créer le dictionnaire de résultats
        results = {
            'score': float(np.mean(global_scores)) if global_scores else 0.0,
            'individual_scores': individual_scores,
            'aspects': aspect_scores,
            'task': self.task
        }
        
        return results