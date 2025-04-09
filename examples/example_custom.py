"""
Exemple de création et d'utilisation d'une métrique personnalisée.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metric_interface import TextMetric
from core.evaluator import TextEvaluator


class JaccardSimilarityMetric(TextMetric):
    """
    Métrique basée sur l'indice de similarité de Jaccard.
    
    L'indice de Jaccard mesure la similarité entre deux ensembles en 
    divisant la taille de leur intersection par la taille de leur union.
    """
    
    def __init__(self, tokenize: bool = True, lowercase: bool = True):
        """
        Initialise la métrique de similarité de Jaccard.
        
        Args:
            tokenize: Tokeniser les textes en mots
            lowercase: Convertir les textes en minuscules
        """
        self.tokenize = tokenize
        self.lowercase = lowercase
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "jaccard"
    
    @property
    def requires_tokenization(self) -> bool:
        """Indique si la métrique nécessite une tokenisation spécifique."""
        return self.tokenize
    
    def preprocess(self, texts: List[str]) -> List[Any]:
        """
        Prétraite les textes pour la métrique.
        
        Args:
            texts: Liste de textes à prétraiter
            
        Returns:
            Liste de textes prétraités (ensemble de mots)
        """
        try: 
            if isinstance(texts[0], set):
                # Si les textes sont déjà des ensembles, on ne fait rien
                return texts
            result = []
            for text in texts:
                if self.lowercase:
                    text = text.lower()
                
                if self.tokenize:
                    # Tokenisation simple par espaces
                    tokens = text.split()
                    # Convertir en ensemble pour le calcul de Jaccard
                    result.append(set(tokens))
                else:
                    # Utiliser les caractères comme tokens
                    result.append(set(text))
                    
            return result
        except Exception as e:
            print(f"Erreur lors du prétraitement des textes : {e}")
            print("Input was:", texts)
            return []
        
    def compute(self, references: List[Any], candidates: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Calcule la similarité de Jaccard entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence (prétraités ou non)
            candidates: Liste de textes candidats (prétraités ou non)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        # Prétraiter si nécessaire
        if self.requires_tokenization and not kwargs.get('already_preprocessed', False):
            refs = self.preprocess(references)
            cands = self.preprocess(candidates)
        else:
            refs = references
            cands = candidates
            
        individual_scores = []
        
        for ref_set, cand_set in zip(refs, cands):
            # Calculer l'indice de Jaccard
            intersection = len(ref_set.intersection(cand_set))
            union = len(ref_set.union(cand_set))
            
            # Éviter la division par zéro
            if union == 0:
                score = 0
            else:
                score = intersection / union
                
            individual_scores.append(score)
            
        return {
            'score': np.mean(individual_scores),
            'individual_scores': individual_scores
        }


def main():
    # Création de l'évaluateur
    evaluator = TextEvaluator()
    
    # Ajouter notre métrique personnalisée
    evaluator.add_metric(JaccardSimilarityMetric())
    
    # Textes à évaluer
    references = [
        "The cat is on the mat.",
        "The weather is nice today.",
        "I love reading books."
    ]
    candidates = [
        "There is a cat on the mat.",
        "We have good weather today.",
        "Books are my favorite thing to read."
    ]
    
    # Évaluation
    results = evaluator.evaluate(references, candidates)
    
    # Affichage des résultats
    print("\nRésultats de l'évaluation:")
    for metric_name, metric_results in results.items():
        print(f"\n{metric_name}:")
        print(f"  Score global: {metric_results['score']:.4f}")
        print(f"  Scores individuels: {[round(s, 4) for s in metric_results['individual_scores']]}")
    
    # Évaluation avec des scores humains simulés
    human_scores = [0.7, 0.8, 0.5]
    
    correlation_results = evaluator.evaluate_with_human_correlation(
        references,
        candidates,
        human_scores
    )
    
    print("\nRapport de corrélation avec les évaluations humaines:")
    print(correlation_results['correlation_report'])
    
    # Génération d'un rapport
    report = evaluator.generate_report(correlation_results)
    print("\nRésumé du rapport:")
    print(report.summary())


if __name__ == "__main__":
    main()
