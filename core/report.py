"""
Module pour la génération de rapports d'évaluation.
"""

import pandas as pd
import json
import yaml
from typing import Dict, List, Any, Optional
import os
from datetime import datetime


class Report:
    """Classe pour gérer la génération et l'exportation des rapports d'évaluation."""
    
    def __init__(self, results: Optional[Dict[str, Any]] = None):
        """
        Initialise le rapport avec des résultats optionnels.
        
        Args:
            results: Résultats d'évaluation (optionnel)
        """
        self.results = results or {}
        self.timestamp = datetime.now()
        
    def add_results(self, results: Dict[str, Any]) -> None:
        """
        Ajoute des résultats au rapport.
        
        Args:
            results: Nouveaux résultats à ajouter
        """
        self.results.update(results)
        
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convertit les résultats en DataFrame pour faciliter l'analyse.
        
        Returns:
            pd.DataFrame: DataFrame contenant les scores globaux des métriques
        """
        if not self.results or 'evaluation_results' not in self.results:
            return pd.DataFrame()
            
        data = []
        for metric_name, metric_results in self.results['evaluation_results'].items():
            if 'score' in metric_results:
                data.append({
                    'Metric': metric_name,
                    'Score': metric_results['score']
                })
                
        return pd.DataFrame(data)
    
    def save(self, output_dir: str, prefix: str = 'eval_report') -> Dict[str, str]:
        """
        Sauvegarde le rapport dans plusieurs formats.
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe pour les noms de fichiers
            
        Returns:
            Dict: Chemins des fichiers générés
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{prefix}_{timestamp_str}"
        
        saved_files = {}
        
        # Fonction pour convertir les DataFrames en dictionnaires
        def convert_dataframes(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, (list, tuple)):
                return [convert_dataframes(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataframes(v) for k, v in obj.items()}
            else:
                return obj
        
        # Convertir tous les DataFrames en dictionnaires pour la sauvegarde JSON
        json_safe_results = convert_dataframes(self.results)
        
        # Sauvegarde en JSON
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
        saved_files['json'] = json_path
        
        # Sauvegarde en YAML
        yaml_path = os.path.join(output_dir, f"{base_filename}.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(json_safe_results, f, default_flow_style=False, allow_unicode=True)
        saved_files['yaml'] = yaml_path
        
        # Sauvegarde en CSV (uniquement les résultats principaux)
        if 'evaluation_results' in self.results:
            csv_path = os.path.join(output_dir, f"{base_filename}_metrics.csv")
            self.to_dataframe().to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
            
        # Sauvegarde du rapport de corrélation si présent
        if 'correlation_report' in self.results and isinstance(self.results['correlation_report'], pd.DataFrame):
            corr_csv_path = os.path.join(output_dir, f"{base_filename}_correlations.csv")
            self.results['correlation_report'].to_csv(corr_csv_path, index=False)
            saved_files['correlation_csv'] = corr_csv_path
            
        return saved_files
    
    def summary(self) -> str:
        """
        Génère un résumé textuel des résultats.
        
        Returns:
            str: Résumé formaté
        """
        if not self.results:
            return "Aucun résultat disponible."
            
        lines = ["# Résumé de l'évaluation", ""]
        
        # Résultats des métriques
        if 'evaluation_results' in self.results:
            lines.append("## Scores des métriques")
            for metric, results in self.results['evaluation_results'].items():
                if 'score' in results:
                    lines.append(f"- {metric}: {results['score']:.4f}")
            lines.append("")
            
        # Résultats de corrélation
        if 'correlations' in self.results:
            lines.append("## Corrélations avec les évaluations humaines")
            for metric, corrs in self.results['correlations'].items():
                corr_str = ", ".join([f"{method}={value:.4f}" for method, value in corrs.items() if value is not None])
                lines.append(f"- {metric}: {corr_str}")
            lines.append("")
            
            # Meilleure métrique
            if 'best_metric' in self.results:
                metric, value = self.results['best_metric']
                lines.append(f"## Meilleure métrique: {metric} ({value:.4f})")
                
        return "\n".join(lines)
