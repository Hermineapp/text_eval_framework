"""
Module implémentant la métrique QuestEval pour l'évaluation de texte.
"""

import os
import sys
import importlib.util
import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

# Déterminer le chemin absolu du répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))

# Définir le chemin vers questeval_lib (créer ce répertoire s'il n'existe pas)
questeval_lib_dir = os.path.join(current_dir, 'questeval_lib')
os.makedirs(questeval_lib_dir, exist_ok=True)

# Créer un __init__.py dans le répertoire questeval_lib s'il n'existe pas
init_file = os.path.join(questeval_lib_dir, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('# Questeval library\n')

# Vérifier si questeval_metric.py et utils.py existent
questeval_metric_file = os.path.join(questeval_lib_dir, 'questeval_metric.py')
utils_file = os.path.join(questeval_lib_dir, 'utils.py')

_QUESTEVAL_AVAILABLE = False
QuestEvalMetricImpl = None


from .questeval_lib.questeval_metric import QuestEval as QuestEvalMetricImpl


# Essayer d'importer QuestEval
try:
    # D'abord essayer l'importation normale
    try:
        from .questeval_lib.questeval_metric import QuestEval as QuestEvalMetricImpl
        _QUESTEVAL_AVAILABLE = True
    except ImportError:
        # Ensuite essayer d'importer depuis le package questeval s'il est installé
        try:
            from questeval.questeval_metric import QuestEval as QuestEvalMetricImpl
            _QUESTEVAL_AVAILABLE = True
        except ImportError:
            logging.warning("QuestEval n'a pas pu être importé. Assurez-vous que le package est installé.")
            _QUESTEVAL_AVAILABLE = False
except Exception as e:
    logging.error(f"Erreur lors de l'importation de QuestEval: {e}")
    _QUESTEVAL_AVAILABLE = False


class QuestEvalMetric(TextMetric):
    """
    Implémentation de la métrique QuestEval pour l'évaluation de texte.
    
    QuestEval est une métrique qui évalue la qualité d'un texte généré en utilisant
    des modèles de question-réponse et de génération de questions pour comparer
    les textes candidats avec les références.
    """
    
    def __init__(self, task: str = "summarization", language: str = "en", 
                no_cuda: bool = False, batch_size: int = 8, 
                use_progress_bar: bool = False, max_questions_per_doc: int = 20,
                qg_batch_size: int = 36, clf_batch_size: int = 48,
                do_weighter: bool = False, do_consistency: bool = False):
        """
        Initialise la métrique QuestEval.
        
        Args:
            task: Tâche d'évaluation ("summarization", "text_generation", etc.)
            language: Langue des textes ("en" ou "fr")
            no_cuda: Si True, utilise le CPU au lieu du GPU
            batch_size: Taille du batch pour les inférences
            use_progress_bar: Afficher une barre de progression pendant l'évaluation
            max_questions_per_doc: Nombre maximum de questions par document
            
        Raises:
            ImportError: Si le package 'questeval' n'est pas installé
        """
        if not _QUESTEVAL_AVAILABLE:
            raise ImportError(
                "Le package 'questeval' est requis pour utiliser QuestEvalMetric. "
                "Installez-le avec 'pip install questeval'."
            )
        
        # Créer le répertoire de logs si nécessaire
        log_dir = os.path.join(questeval_lib_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        self.task = task
        self.language = language
        self.no_cuda = no_cuda
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.use_progress_bar = use_progress_bar
        self.max_questions_per_doc = max_questions_per_doc
        self.do_weighter = do_weighter
        self.do_consistency = do_consistency
        
        # Initialiser la métrique QuestEval
        self.questeval = QuestEvalMetricImpl(
            task=task,
            language=language,
            no_cuda=no_cuda,
            qg_batch_size=qg_batch_size,
            clf_batch_size=clf_batch_size,
            #use_progress_bar=use_progress_bar,
            #max_questions_per_doc=max_questions_per_doc,
            do_weighter=do_weighter,
            do_consistency=do_consistency
        )
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "questeval"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores QuestEval entre les références et les candidats.
        
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
        
        # Adapter les entrées au format attendu par QuestEval
        if self.task == "summarization":
            # Pour la tâche de résumé, QuestEval attend des références
            result = self.questeval.corpus_questeval(
                hypothesis=candidates,
                sources=None,
                list_references=[[ref] for ref in references]
            )
        else:
            # Pour d'autres tâches comme la traduction ou la génération de texte,
            # QuestEval attend des sources
            result = self.questeval.corpus_questeval(
                hypothesis=candidates,
                sources=references,
                list_references=None
            )
        
        # Extraire les scores individuels
        individual_scores = result.get('ex_level_scores', [])
        
        # Calculer le score global (moyenne)
        global_score = result.get('corpus_score', np.mean(individual_scores) if individual_scores else 0)
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'task': self.task,
            'language': self.language
        }
