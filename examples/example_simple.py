"""
Exemple simple d'utilisation du framework d'évaluation de texte.
"""

import sys
import os

# Ajouter le répertoire parent au chemin de recherche
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import direct des modules
from core.evaluator import TextEvaluator
from metrics.bleu import BLEUMetric
from metrics.rouge import ROUGEMetric


def main():
    # Création de l'évaluateur
    evaluator = TextEvaluator()

    # Ajout de métriques
    evaluator.add_metric(BLEUMetric())
    
    try:
        evaluator.add_metric(ROUGEMetric())
        print("La métrique ROUGE a été chargée avec succès.")
    except ImportError:
        print("La métrique ROUGE n'a pas pu être chargée. Vérifiez que les dépendances sont installées.")
    except Exception as e:
        print(f"Erreur lors du chargement de la métrique ROUGE: {e}")

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

    # Génération d'un rapport
    report = evaluator.generate_report(results)
    print("\nRésumé du rapport:")
    print(report.summary())

    # Sauvegarde du rapport
    output_dir = os.path.join(parent_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = report.save(output_dir)
    print(f"\nRapport sauvegardé dans: {', '.join(saved_files.values())}")


if __name__ == "__main__":
    main()