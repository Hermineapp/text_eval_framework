"""
Module pour la corrélation des scores automatiques avec les évaluations humaines.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
from typing import Dict, List, Union, Callable, Optional


class MetricCorrelation:
    """Évalue la corrélation entre les scores métriques et les évaluations humaines."""
    
    def __init__(self):
        """Initialise l'objet avec les méthodes de corrélation standard."""
        self.correlation_methods = {
            'pearson': lambda x, y: pearsonr(x, y)[0],
            'spearman': lambda x, y: spearmanr(x, y)[0],
            'kendall': lambda x, y: kendalltau(x, y)[0]
        }
        
    def add_correlation_method(self, name: str, method: Callable[[List, List], float]) -> None:
        """
        Ajoute une méthode de corrélation personnalisée.
        
        Args:
            name: Nom de la méthode
            method: Fonction qui calcule la corrélation entre deux listes
        """
        self.correlation_methods[name] = method
        
    def compute_correlations(self, 
                           metric_scores: Dict[str, List[float]], 
                           human_scores: List[float],
                           methods: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calcule les corrélations entre les scores des métriques et les évaluations humaines.
        
        Args:
            metric_scores: Dictionnaire {nom_métrique: [scores]} pour chaque instance
            human_scores: Liste des scores humains pour chaque instance
            methods: Liste des méthodes de corrélation à utiliser (None = toutes)
            
        Returns:
            Dict: {nom_métrique: {méthode_corrélation: score}}
        """
        if methods is None:
            methods_to_use = self.correlation_methods
        else:
            methods_to_use = {k: self.correlation_methods[k] for k in methods if k in self.correlation_methods}
            
        results = {}
        
        for metric_name, scores in metric_scores.items():
            if len(scores) != len(human_scores):
                raise ValueError(
                    f"Le nombre de scores pour {metric_name} ({len(scores)}) ne correspond pas "
                    f"au nombre de scores humains ({len(human_scores)})"
                )
                
            metric_results = {}
            for method_name, correlation_func in methods_to_use.items():
                try:
                    correlation = correlation_func(scores, human_scores)
                    metric_results[method_name] = correlation
                except Exception as e:
                    metric_results[method_name] = None
                    print(f"Erreur lors du calcul de {method_name} pour {metric_name}: {e}")
                    
            results[metric_name] = metric_results
            
        return results
    
    def generate_correlation_report(self, correlations: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Génère un rapport de corrélation sous forme de DataFrame.
        
        Args:
            correlations: Résultat de compute_correlations
            
        Returns:
            pd.DataFrame: Tableau des corrélations
        """
        metrics = list(correlations.keys())
        methods = list(self.correlation_methods.keys())
        
        data = []
        for metric in metrics:
            row = [metric]
            for method in methods:
                value = correlations.get(metric, {}).get(method)
                row.append(value)
            data.append(row)
            
        columns = ['Metric'] + list(methods)
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def find_best_metric(self, correlations: Dict[str, Dict[str, float]], 
                        method: str = 'pearson') -> tuple:
        """
        Identifie la métrique la plus corrélée avec les jugements humains.
        
        Args:
            correlations: Résultat de compute_correlations
            method: Méthode de corrélation à utiliser
            
        Returns:
            tuple: (nom_de_la_métrique, valeur_de_corrélation)
        """
        best_metric = None
        best_correlation = -float('inf')
        
        for metric, metric_correlations in correlations.items():
            if method in metric_correlations and metric_correlations[method] is not None:
                if metric_correlations[method] > best_correlation:
                    best_correlation = metric_correlations[method]
                    best_metric = metric
                    
        return best_metric, best_correlation
