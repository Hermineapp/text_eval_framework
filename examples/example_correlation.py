"""
Exemple d'utilisation du framework avec corrélation aux évaluations humaines.
"""

import sys
import os

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.evaluator import TextEvaluator
from metrics.bleu import BLEUMetric
from metrics.rouge import ROUGEMetric
from utils.visualization import plot_metric_correlations, plot_metric_vs_human


def main():
    # Création de l'évaluateur
    evaluator = TextEvaluator()

    # Ajout de métriques
    evaluator.add_metric(BLEUMetric())
    evaluator.add_metric(ROUGEMetric())

    # Charger les données
    references = [
        "The cat is on the mat.",
        "The weather is nice today.",
        "I love reading books.",
        "Paris is the capital of France.",
        "The Eiffel Tower is in Paris."
    ]
    candidates = [
        "There is a cat on the mat.",
        "We have good weather today.",
        "Books are my favorite thing to read.",
        "Paris serves as France's capital city.",
        "The Eiffel Tower can be found in Paris."
    ]

    # Charger les scores humains (1-5, où 5 est la meilleure qualité)
    human_scores = [4.2, 3.8, 3.5, 4.5, 4.0]

    # Évaluer et calculer les corrélations
    results = evaluator.evaluate_with_human_correlation(
        references, 
        candidates,
        human_scores,
        correlation_methods=['pearson', 'spearman']
    )

    # Afficher les résultats de l'évaluation
    print("Résultats de l'évaluation des métriques:")
    for metric_name, metric_results in results['evaluation_results'].items():
        print(f"\n{metric_name}:")
        print(f"  Score global: {metric_results['score']:.4f}")
        print(f"  Scores individuels: {[round(s, 4) for s in metric_results['individual_scores']]}")

    # Afficher le rapport de corrélation
    print("\nRapport de corrélation avec les évaluations humaines:")
    print(results['correlation_report'])

    # Identifier la métrique la plus corrélée avec les jugements humains
    best_metric, best_correlation = results['best_metric']
    print(f"\nLa métrique la plus corrélée avec les jugements humains: {best_metric} ({best_correlation:.3f})")

    # Visualisations
    output_dir = "correlation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer une heatmap des corrélations
    plot_metric_correlations(
        results['correlation_report'],
        save_path=os.path.join(output_dir, "correlation_heatmap.png")
    )
    print(f"\nHeatmap des corrélations sauvegardée dans: {os.path.join(output_dir, 'correlation_heatmap.png')}")
    
    # Créer des visualisations pour chaque métrique
    for metric, metric_results in results['evaluation_results'].items():
        if 'individual_scores' in metric_results:
            for method, corr_value in results['correlations'][metric].items():
                plot_metric_vs_human(
                    metric_results['individual_scores'],
                    human_scores,
                    metric,
                    corr_value,
                    correlation_type=method.capitalize(),
                    save_path=os.path.join(output_dir, f"{metric}_{method}_correlation.png")
                )
    print(f"Visualisations des corrélations sauvegardées dans: {output_dir}")
    
    # Génération d'un rapport
    report = evaluator.generate_report(results, output_dir)
    print("\nRésumé du rapport:")
    print(report.summary())


if __name__ == "__main__":
    main()
