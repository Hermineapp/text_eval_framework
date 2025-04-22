"""
Module principal contenant la classe TextEvaluator qui gère l'évaluation des textes.
"""

from typing import List, Dict, Any, Type, Optional, Union
import importlib
import json
import yaml
import pandas as pd
import os
from .metric_interface import TextMetric
from .correlation import MetricCorrelation
from .report import Report


class TextEvaluator:
    """Classe principale du framework d'évaluation."""
    
    def __init__(self):
        """Initialise l'évaluateur avec un dictionnaire vide de métriques."""
        self.metrics = {}
        self.correlation = MetricCorrelation()
        self.visualization_config = {}
        
    def add_metric(self, metric: TextMetric) -> None:
        """
        Ajoute une métrique à l'évaluateur.
        
        Args:
            metric: Instance d'une classe implémentant TextMetric
        """
        self.metrics[metric.name] = metric
        
    def load_metric(self, metric_path: str, **kwargs) -> None:
        """
        Charge dynamiquement une métrique depuis un module.
        
        Args:
            metric_path: Chemin du module et nom de la classe (ex: "metrics.bleu.BLEUMetric")
            **kwargs: Paramètres à passer au constructeur de la métrique
        
        Raises:
            ValueError: Si la métrique ne peut pas être chargée
        """
        try:
            # Gérer les chemins absolus et relatifs
            if metric_path.startswith("text_eval_framework."):
                # Format absolu (après installation du package)
                module_path, class_name = metric_path.rsplit('.', 1)
                try:
                    module = importlib.import_module(module_path)
                except ImportError:
                    # Essayer avec le chemin relatif si le chemin absolu échoue
                    relative_path = '.'.join(module_path.split('.')[1:])
                    module = importlib.import_module(relative_path)
            else:
                # Format relatif (dans le développement)
                module_path, class_name = metric_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                
            metric_class = getattr(module, class_name)
            metric = metric_class(**kwargs)
            self.add_metric(metric)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Impossible de charger la métrique {metric_path}: {e}")
            
    def evaluate(self, references: List[str], candidates: List[str], 
                metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Évalue les textes candidats par rapport aux références.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats à évaluer
            metrics: Liste des noms de métriques à utiliser (None = toutes)
            
        Returns:
            dict: Résultats pour chaque métrique
        """
        if metrics is None:
            metrics_to_use = self.metrics
        else:
            metrics_to_use = {k: self.metrics[k] for k in metrics if k in self.metrics}
            
        if not metrics_to_use:
            raise ValueError("Aucune métrique disponible pour l'évaluation.")
            
        results = {}
        for name, metric in metrics_to_use.items():
            refs = metric.preprocess(references) if hasattr(metric, 'preprocess') else references
            cands = metric.preprocess(candidates) if hasattr(metric, 'preprocess') else candidates
            metric_results = metric.compute(refs, cands)
            results[name] = metric_results
            
            # Ajout des métriques dérivées pour ROUGE (précision et rappel)
            if name == "rouge" and 'precision' in metric_results and 'recall' in metric_results:
                # Ajouter les métriques dérivées comme des métriques distinctes
                results["rouge_precision"] = metric_results["precision"]
                results["rouge_recall"] = metric_results["recall"]
                results["rouge_f1"] = metric_results["f1"]
            
        return results
    
    def evaluate_with_human_correlation(self, 
                                      references: List[str], 
                                      candidates: List[str],
                                      human_scores: List[float],
                                      metrics: Optional[List[str]] = None,
                                      correlation_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Évalue les textes et calcule les corrélations avec les scores humains.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            human_scores: Liste des scores d'évaluation humaine
            metrics: Liste des métriques à utiliser (None = toutes)
            correlation_methods: Liste des méthodes de corrélation à utiliser
            
        Returns:
            dict: Résultats d'évaluation et corrélations
        """
        # Vérifier que les dimensions sont correctes
        if len(candidates) != len(human_scores):
            raise ValueError(
                f"Le nombre de candidats ({len(candidates)}) ne correspond pas "
                f"au nombre de scores humains ({len(human_scores)})"
            )
            
        # Évaluer avec toutes les métriques
        eval_results = self.evaluate(references, candidates, metrics)
        
        # Extraire les scores individuels pour chaque métrique
        metric_scores = {}
        for metric_name, result in eval_results.items():
            if 'individual_scores' in result:
                metric_scores[metric_name] = result['individual_scores']
            
        # Calculer les corrélations
        correlations = self.correlation.compute_correlations(
            metric_scores, 
            human_scores,
            correlation_methods
        )
        
        # Rapport de corrélation
        correlation_report = self.correlation.generate_correlation_report(correlations)
        
        # Déterminer la meilleure métrique
        best_metric = self.correlation.find_best_metric(correlations, 'spearman')
        
        return {
            'evaluation_results': eval_results,
            'correlations': correlations,
            'correlation_report': correlation_report,
            'best_metric': best_metric
        }
    
    def from_config(self, config_path: str) -> None:
        """
        Configure l'évaluateur à partir d'un fichier YAML ou JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Raises:
            ValueError: Si le format de configuration n'est pas supporté
        """
        ext = config_path.split('.')[-1].lower()
        
        with open(config_path, 'r') as f:
            if ext == 'json':
                config = json.load(f)
            elif ext in ('yaml', 'yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Format de configuration non supporté: {ext}")
        
        # Charger les métriques
        for metric_config in config.get('metrics', []):
            name = metric_config.pop('name')
            params = metric_config.pop('params', {})
            self.load_metric(name, **params)
        
        # Charger la configuration de corrélation si présente
        if 'correlation' in config:
            corr_config = config['correlation']
            
            # Configuration des méthodes de corrélation
            if 'methods' in corr_config:
                # Vérification que toutes les méthodes sont supportées
                for method in corr_config['methods']:
                    if method not in self.correlation.correlation_methods:
                        print(f"Avertissement: méthode de corrélation non supportée: {method}")
            
            # Configuration des chemins de données
            if 'data' in corr_config and 'human_scores_path' in corr_config['data']:
                # Stocker les chemins pour utilisation ultérieure
                self.human_scores_path = corr_config['data']['human_scores_path']
                
                # Optionnel: stocker les chemins vers les données de texte
                if 'references_path' in corr_config['data']:
                    self.references_path = corr_config['data']['references_path']
                if 'candidates_path' in corr_config['data']:
                    self.candidates_path = corr_config['data']['candidates_path']
            
            # Configuration de la visualisation
            if 'visualization' in corr_config:
                self.visualization_config = corr_config['visualization']
    
    def load_human_scores(self, filepath: str) -> List[float]:
        """
        Charge les scores humains depuis un fichier CSV ou JSON.
        
        Args:
            filepath: Chemin vers le fichier de scores
            
        Returns:
            List[float]: Liste des scores humains
            
        Raises:
            ValueError: Si le format du fichier n'est pas reconnu
        """
        ext = filepath.split('.')[-1].lower()
        
        if ext == 'csv':
            df = pd.read_csv(filepath)
            # Essaie de détecter automatiquement la colonne contenant les scores
            score_columns = [col for col in df.columns 
                           if 'score' in col.lower() or 'human' in col.lower() or 'eval' in col.lower()]
            
            if score_columns:
                return df[score_columns[0]].tolist()
            else:
                # Si aucune colonne évidente, prend la dernière colonne
                return df.iloc[:, -1].tolist()
                
        elif ext == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Tente de trouver les scores dans différentes structures JSON possibles
            if isinstance(data, list):
                if all(isinstance(item, (int, float)) for item in data):
                    return data
                elif all(isinstance(item, dict) for item in data):
                    # Cherche une clé de score
                    potential_keys = ['score', 'human_score', 'evaluation', 'rating']
                    for key in potential_keys:
                        if key in data[0]:
                            return [item[key] for item in data]
            
            elif isinstance(data, dict):
                potential_keys = ['scores', 'human_scores', 'evaluations', 'ratings']
                for key in potential_keys:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                        
        raise ValueError(f"Impossible de charger les scores humains depuis {filepath}. Format non reconnu.")
    
    def generate_report(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> Report:
        """
        Génère un rapport à partir des résultats d'évaluation.
        
        Args:
            results: Résultats d'évaluation
            output_dir: Répertoire de sortie pour sauvegarder le rapport (optionnel)
            
        Returns:
            Report: Objet rapport
        """
        report = Report(results)
        
        if output_dir:
            report.save(output_dir)
            
        return report