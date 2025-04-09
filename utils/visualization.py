"""
Module pour la visualisation des résultats d'évaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import os


def plot_metric_correlations(correlation_report: pd.DataFrame, 
                           figsize: tuple = (10, 6),
                           cmap: str = 'viridis',
                           save_path: Optional[str] = None) -> None:
    """
    Visualise les corrélations des métriques avec les évaluations humaines.
    
    Args:
        correlation_report: DataFrame contenant les corrélations
        figsize: Taille de la figure
        cmap: Palette de couleurs
        save_path: Chemin pour sauvegarder l'image (None = afficher)
    """
    plt.figure(figsize=figsize)
    
    # Préparer les données
    df = correlation_report.set_index('Metric')
    
    # Créer un heatmap
    sns.heatmap(df, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, fmt='.3f')
    plt.title('Corrélation des métriques avec les évaluations humaines')
    plt.tight_layout()
    
    if save_path:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
        
def plot_metric_vs_human(metric_scores: List[float],
                       human_scores: List[float],
                       metric_name: str,
                       correlation: float,
                       correlation_type: str = 'Pearson',
                       figsize: tuple = (8, 6),
                       save_path: Optional[str] = None) -> None:
    """
    Crée un nuage de points comparant les scores d'une métrique aux scores humains.
    
    Args:
        metric_scores: Scores de la métrique
        human_scores: Scores humains correspondants
        metric_name: Nom de la métrique
        correlation: Valeur de corrélation à afficher
        correlation_type: Type de corrélation (Pearson, Spearman, etc.)
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder l'image (None = afficher)
    """
    plt.figure(figsize=figsize)
    
    # Créer le nuage de points
    plt.scatter(human_scores, metric_scores, alpha=0.7)
    
    # Ajouter une ligne de tendance
    z = np.polyfit(human_scores, metric_scores, 1)
    p = np.poly1d(z)
    plt.plot(human_scores, p(human_scores), "r--", alpha=0.7)
    
    plt.xlabel("Scores humains")
    plt.ylabel(f"Scores {metric_name}")
    plt.title(f"{metric_name} vs. Évaluations humaines\n{correlation_type} r = {correlation:.3f}")
    plt.grid(alpha=0.3)
    
    if save_path:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_metric_comparison(metric_results: Dict[str, Dict[str, Union[float, List[float]]]],
                         figsize: tuple = (12, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Crée un graphique comparant les scores de différentes métriques.
    
    Args:
        metric_results: Résultats de l'évaluation
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder l'image (None = afficher)
    """
    plt.figure(figsize=figsize)
    
    # Extraire les noms et scores des métriques
    metrics = list(metric_results.keys())
    scores = [results['score'] for results in metric_results.values()]
    
    # Créer un graphique à barres
    bars = plt.bar(metrics, scores, alpha=0.7)
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel("Métriques")
    plt.ylabel("Score moyen")
    plt.title("Comparaison des scores des différentes métriques")
    plt.ylim(0, max(scores) * 1.1)  # Ajuster l'échelle y
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_metric_distribution(metric_name: str,
                           individual_scores: List[float],
                           bins: int = 10,
                           figsize: tuple = (8, 6),
                           save_path: Optional[str] = None) -> None:
    """
    Crée un histogramme montrant la distribution des scores individuels d'une métrique.
    
    Args:
        metric_name: Nom de la métrique
        individual_scores: Scores individuels
        bins: Nombre de bins pour l'histogramme
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder l'image (None = afficher)
    """
    plt.figure(figsize=figsize)
    
    # Créer l'histogramme
    plt.hist(individual_scores, bins=bins, alpha=0.7)
    
    # Ajouter une ligne pour la moyenne
    mean_score = np.mean(individual_scores)
    plt.axvline(mean_score, color='r', linestyle='dashed', linewidth=2)
    plt.text(mean_score, plt.ylim()[1]*0.9, f'Moyenne: {mean_score:.3f}',
            horizontalalignment='center', color='r')
    
    plt.xlabel(f"Scores {metric_name}")
    plt.ylabel("Fréquence")
    plt.title(f"Distribution des scores de la métrique {metric_name}")
    plt.grid(alpha=0.3)
    
    if save_path:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_correlation_dashboard(correlation_results: Dict[str, Any],
                               output_dir: str,
                               prefix: str = "correlation") -> List[str]:
    """
    Crée un ensemble de visualisations pour les corrélations des métriques.
    
    Args:
        correlation_results: Résultats de l'évaluation avec corrélations
        output_dir: Répertoire de sortie
        prefix: Préfixe pour les noms de fichiers
        
    Returns:
        List: Liste des chemins des fichiers générés
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # Heatmap de corrélation
    heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
    plot_metric_correlations(
        correlation_results['correlation_report'],
        save_path=heatmap_path
    )
    generated_files.append(heatmap_path)
    
    # Nuages de points pour chaque métrique
    for metric, metric_results in correlation_results['evaluation_results'].items():
        if 'individual_scores' in metric_results:
            for method, corr_value in correlation_results['correlations'][metric].items():
                if corr_value is not None:
                    scatter_path = os.path.join(output_dir, f"{prefix}_{metric}_{method}.png")
                    plot_metric_vs_human(
                        metric_results['individual_scores'],
                        correlation_results.get('human_scores', []),
                        metric,
                        corr_value,
                        correlation_type=method.capitalize(),
                        save_path=scatter_path
                    )
                    generated_files.append(scatter_path)
    
    # Distribution des scores humains
    if 'human_scores' in correlation_results:
        human_dist_path = os.path.join(output_dir, f"{prefix}_human_distribution.png")
        plot_metric_distribution(
            "évaluations humaines",
            correlation_results['human_scores'],
            save_path=human_dist_path
        )
        generated_files.append(human_dist_path)
    
    return generated_files
