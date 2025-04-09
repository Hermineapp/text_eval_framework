"""
Exemple complet d'utilisation du framework avec configuration YAML et visualisations.
"""

import sys
import os
import argparse
import pandas as pd

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.evaluator import TextEvaluator
from utils.data_loader import load_eval_data
from utils.config import load_config, validate_config, save_config, create_default_config
from utils.visualization import create_correlation_dashboard


def create_example_config(output_path: str) -> None:
    """
    Crée un fichier de configuration d'exemple.
    
    Args:
        output_path: Chemin de sortie pour le fichier de configuration
    """
    config = create_default_config()
    save_config(config, output_path)
    print(f"Configuration d'exemple créée et sauvegardée dans: {output_path}")


def create_example_data(output_dir: str) -> None:
    """
    Crée des fichiers de données d'exemple.
    
    Args:
        output_dir: Répertoire de sortie pour les fichiers de données
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer des données d'exemple
    data = {
        'reference': [
            "The cat is on the mat.",
            "The weather is nice today.",
            "I love reading books.",
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris."
        ],
        'candidate': [
            "There is a cat on the mat.",
            "We have good weather today.",
            "Books are my favorite thing to read.",
            "Paris serves as France's capital city.",
            "The Eiffel Tower can be found in Paris."
        ],
        'human_score': [4.2, 3.8, 3.5, 4.5, 4.0]
    }
    
    # Sauvegarder en CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "example_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Sauvegarder les références et candidats séparément
    with open(os.path.join(output_dir, "references.txt"), 'w') as f:
        f.write('\n'.join(data['reference']))
    
    with open(os.path.join(output_dir, "candidates.txt"), 'w') as f:
        f.write('\n'.join(data['candidate']))
    
    # Sauvegarder les scores humains séparément
    human_scores_df = pd.DataFrame({'score': data['human_score']})
    human_scores_df.to_csv(os.path.join(output_dir, "human_scores.csv"), index=False)
    
    print(f"Données d'exemple créées et sauvegardées dans: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Exemple complet du framework d'évaluation de texte")
    parser.add_argument("--config", type=str, help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("--data", type=str, help="Chemin vers le fichier de données CSV")
    parser.add_argument("--human-scores", type=str, help="Chemin vers le fichier de scores humains")
    parser.add_argument("--output", type=str, default="results", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--create-config", type=str, help="Créer une configuration d'exemple et l'enregistrer")
    parser.add_argument("--create-data", type=str, help="Créer des données d'exemple et les enregistrer")
    
    args = parser.parse_args()
    
    # Gérer les arguments spéciaux
    if args.create_config:
        create_example_config(args.create_config)
        return
    
    if args.create_data:
        create_example_data(args.create_data)
        return
    
    # Créer l'évaluateur
    evaluator = TextEvaluator()
    
    # Configurer l'évaluateur depuis un fichier
    if args.config:
        print(f"Chargement de la configuration depuis: {args.config}")
        try:
            evaluator.from_config(args.config)
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
            return
    else:
        # Configuration minimale par défaut
        from metrics.bleu import BLEUMetric
        from metrics.rouge import ROUGEMetric
        
        print("Utilisation de la configuration par défaut (BLEU et ROUGE)")
        evaluator.add_metric(BLEUMetric())
        evaluator.add_metric(ROUGEMetric())
    
    # Charger les données
    if args.data:
        print(f"Chargement des données depuis: {args.data}")
        try:
            data = load_eval_data(args.data, args.human_scores)
            references = data['references']
            candidates = data['candidates']
            human_scores = data.get('human_scores')
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return
    else:
        # Données d'exemple
        print("Utilisation des données d'exemple")
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
        human_scores = [4.2, 3.8, 3.5]
    
    # Créer les répertoires de sortie
    os.makedirs(args.output, exist_ok=True)
    visualizations_dir = os.path.join(args.output, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Évaluer et calculer les corrélations si des scores humains sont disponibles
    if human_scores:
        print(f"Évaluation avec corrélation aux scores humains ({len(human_scores)} scores)")
        results = evaluator.evaluate_with_human_correlation(
            references, 
            candidates,
            human_scores
        )
        
        # Ajouter les scores humains aux résultats pour la visualisation
        results['human_scores'] = human_scores
        
        # Créer un tableau de bord de visualisations
        print("Création des visualisations...")
        viz_files = create_correlation_dashboard(results, visualizations_dir)
        print(f"{len(viz_files)} visualisations créées dans: {visualizations_dir}")
        
        # Afficher le rapport de corrélation
        print("\nRapport de corrélation avec les évaluations humaines:")
        print(results['correlation_report'])
        
        # Identifier la meilleure métrique
        best_metric, best_correlation = results['best_metric']
        print(f"\nLa métrique la plus corrélée avec les jugements humains: {best_metric} ({best_correlation:.3f})")
    else:
        print("Évaluation sans corrélation (pas de scores humains disponibles)")
        results = evaluator.evaluate(references, candidates)
    
    # Générer un rapport
    report = evaluator.generate_report(results, args.output)
    
    print("\nRésumé du rapport:")
    print(report.summary())
    print(f"\nRapport complet sauvegardé dans: {args.output}")


if __name__ == "__main__":
    main()
