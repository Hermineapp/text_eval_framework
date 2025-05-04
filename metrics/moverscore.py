"""
Module implémentant la métrique MoverScore pour l'évaluation de texte.

Cette implémentation utilise directement le package moverscore_v2 qui doit être
installé via pip: pip install moverscore
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.metric_interface import TextMetric
from collections import defaultdict

try:
    from moverscore_v2 import word_mover_score
    _MOVERSCORE_AVAILABLE = True
except ImportError:
    _MOVERSCORE_AVAILABLE = False


class MoverScoreMetric(TextMetric):
    """
    Implémentation de la métrique MoverScore pour l'évaluation de texte.

    MoverScore utilise la distance Earth Mover's Distance (EMD) entre les embeddings BERT
    des textes pour calculer leur similarité sémantique.
    """

    def __init__(self,
                n_gram: int = 1,
                remove_subwords: bool = True,
                stop_words: List[str] = None,
                batch_size: int = 48):
        """
        Initialise la métrique MoverScore.

        Args:
            n_gram: Taille des n-grammes à considérer (1 pour mots individuels)
            remove_subwords: Supprimer les sous-mots (commençant par ##)
            stop_words: Liste des mots vides à ignorer
            batch_size: Taille du batch pour le calcul

        Raises:
            ImportError: Si le package 'moverscore_v2' n'est pas installé
        """
        if not _MOVERSCORE_AVAILABLE:
            raise ImportError(
                "Le package 'moverscore_v2' est requis pour utiliser MoverScoreMetric. "
                "Installez-le avec: 'pip install moverscore'"
            )

        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.stop_words = stop_words or []
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "moverscore"

    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores MoverScore entre les références et les candidats.

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

        # Dictionnaires IDF par défaut (avec poids uniforme de 1.0)
        idf_dict_hyp = defaultdict(lambda: 1.0)
        idf_dict_ref = defaultdict(lambda: 1.0)

        # Calculer MoverScore par batch
        individual_scores = []
        for i in range(0, len(references), self.batch_size):
            batch_refs = references[i:i + self.batch_size]
            batch_cands = candidates[i:i + self.batch_size]

            # Pour chaque candidat, nous créons une liste avec la même référence répétée
            # afin de respecter l'interface de la fonction word_mover_score
            scores_batch = []
            for ref, cand in zip(batch_refs, batch_cands):
                scores = word_mover_score(
                    references=[ref],  # Liste avec une seule référence
                    hypotheses=[cand],  # Liste avec un seul candidat
                    idf_dict_ref=idf_dict_ref,
                    idf_dict_hyp=idf_dict_hyp,
                    stop_words=self.stop_words,
                    n_gram=self.n_gram,
                    remove_subwords=self.remove_subwords
                )
                scores_batch.append(np.mean(scores))

            individual_scores.extend(scores_batch)

        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)

        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'n_gram': self.n_gram,
            'remove_subwords': self.remove_subwords
        }