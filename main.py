"""
Script principal pour l'exécution du framework d'évaluation de texte depuis la ligne de commande.
"""

import argparse
import os
import sys
import logging
import json
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional

from .core.evaluator import TextEvaluator
from .utils.data_loader import load_eval_data
from .utils.config import load_config, validate_config, save_config, create_default_config
from .utils.visualization import create_correlation_dashboard
from .metrics import BLEUMetric, ROUGEMetric


def setup_logging(verbose: bool = False) -> None:
    """
    Configure le système de logging.
    
    Args:
        verbose: Activer les messages de débogage
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def create_parser() -> argparse.ArgumentParser:
    """
    Crée le parseur d'arguments pour la ligne de commande.
    
    Returns:
        argparse.ArgumentParser: Parseur configuré
    """
    parser = argparse.ArgumentParser(
        description="""
        Framework d'évaluation de texte - Outil pour évaluer des textes candidats par rapport à des références
        selon différentes métriques, avec corrélation optionnelle aux évaluations humaines.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Arguments principaux
    parser.add_argument(
        '--config', '-c', type=str,
        help='Chemin vers le fichier de configuration YAML ou JSON'
    )
    parser.add_argument(
        '--references', '-r', type=str,
        help='Chemin vers le fichier contenant les textes de référence'
    )
    parser.add_argument(
        '--candidates', '-C', type=str,
        help='Chemin vers le fichier contenant les textes candidats'
    )
    parser.add_argument(
        '--human-scores', '-s', type=str,
        help='Chemin vers le fichier contenant les scores humains'
    )
    parser.add_argument(
        '--metrics', '-m', type=str, nargs='+',
        help='Liste des métriques à utiliser (ex: bleu rouge)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='results',
        help='Répertoire de sortie pour les résultats'
    )
    
    # Options de fonctionnalité
    feature_group = parser.add_argument_group('Fonctionnalités')
    feature_group.add_argument(
        '--create-config', type=str, metavar='FICHIER',
        help="Créer un fichier de configuration d'exemple et l'enregistrer"
    )
    feature_group.add_argument(
        '--list-metrics', action='store_true',
        help='Afficher la liste des métriques disponibles'
    )
    feature_group.add_argument(
        '--visualize', action='store_true',
        help='Générer des visualisations des résultats'
    )
    
    # Options avancées
    advanced_group = parser.add_argument_group('Options avancées')
    advanced_group.add_argument(
        '--correlation-methods', type=str, nargs='+',
        choices=['pearson', 'spearman', 'kendall'],
        default=['pearson', 'spearman'],
        help='Méthodes de corrélation à utiliser'
    )
    advanced_group.add_argument(
        '--save-formats', type=str, nargs='+',
        choices=['json', 'yaml', 'csv'],
        default=['json', 'csv'],
        help='Formats pour la sauvegarde des résultats'
    )
    advanced_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Afficher des informations détaillées pendant l\'exécution'
    )
    
    return parser


def list_available_metrics() -> None:
    """
    Affiche la liste des métriques disponibles et leurs descriptions.
    """
    print("Métriques disponibles:")
    
    metrics = [
        {
            'name': 'bleu',
            'class': 'BLEUMetric',
            'description': 'Bilingual Evaluation Understudy - Évalue la qualité en comparant les n-grammes',
            'options': ['weights', 'smoothing']
        },
        {
            'name': 'rouge',
            'class': 'ROUGEMetric',
            'description': 'Recall-Oriented Understudy for Gisting Evaluation - Évalue la qualité des résumés',
            'options': ['rouge_types', 'use_stemmer']
        },
        {
            'name': 'bert_score',
            'class': 'BERTScoreMetric',
            'description': 'Évalue la qualité en utilisant les embeddings BERT',
            'options': ['model_name', 'batch_size', 'rescale_with_baseline']
        },
        {
            'name': 'meteor',
            'class': 'METEORMetric',
            'description': 'Metric for Evaluation of Translation with Explicit ORdering',
            'options': ['language', 'alpha', 'beta', 'gamma', 'use_synonyms']
        },
        {
            'name': 'sbert',
            'class': 'SentenceBERTMetric',
            'description': 'Évalue la similarité sémantique à l\'aide de SentenceBERT',
            'options': ['model_name', 'batch_size', 'similarity_metric']
        }
    ]
    
    # Format et affichage
    for metric in metrics:
        print(f"\n{metric['name']} ({metric['class']}):")
        print(f"  {metric['description']}")
        print("  Options:")
        for option in metric['options']:
            print(f"    - {option}")


def initialize_evaluator(args: argparse.Namespace) -> TextEvaluator:
    """
    Initialise l'évaluateur en fonction des arguments de la ligne de commande.
    
    Args:
        args: Arguments de la ligne de commande
        
    Returns:
        TextEvaluator: Évaluateur configuré
    """
    evaluator = TextEvaluator()
    
    # Si un fichier de configuration est spécifié, l'utiliser
    if args.config:
        logging.info(f"Chargement de la configuration depuis: {args.config}")
        try:
            evaluator.from_config(args.config)
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {e}")
            sys.exit(1)
    # Sinon, utiliser les métriques spécifiées en ligne de commande ou les métriques par défaut
    else:
        # Métriques par défaut
        if not args.metrics:
            logging.info("Aucune métrique spécifiée, utilisation des métriques par défaut (BLEU, ROUGE)")
            evaluator.add_metric(BLEUMetric())
            evaluator.add_metric(ROUGEMetric())
        else:
            # Métriques spécifiées
            for metric_name in args.metrics:
                metric_name = metric_name.lower()
                try:
                    if metric_name == 'bleu':
                        evaluator.add_metric(BLEUMetric())
                    elif metric_name == 'rouge':
                        evaluator.add_metric(ROUGEMetric())
                    elif metric_name == 'bert_score':
                        from .metrics import BERTScoreMetric
                        evaluator.add_metric(BERTScoreMetric())
                    elif metric_name == 'meteor':
                        from .metrics import METEORMetric
                        evaluator.add_metric(METEORMetric())
                    elif metric_name == 'sbert':
                        from .metrics import SentenceBERTMetric
                        evaluator.add_metric(SentenceBERTMetric())
                    else:
                        logging.warning(f"Métrique inconnue: {metric_name}")
                except ImportError as e:
                    logging.error(f"Impossible de charger la métrique {metric_name}: {e}")
                    logging.error("Assurez-vous que les dépendances nécessaires sont installées.")
                except Exception as e:
                    logging.error(f"Erreur lors du chargement de la métrique {metric_name}: {e}")
    
    return evaluator


def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Charge les données d'évaluation à partir des arguments.
    
    Args:
        args: Arguments de la ligne de commande
        
    Returns:
        Dict: Données chargées (références, candidats, scores humains)
    """
    data = {}
    
    # Charger les références et candidats
    if args.references and args.candidates:
        logging.info(f"Chargement des références depuis: {args.references}")
        logging.info(f"Chargement des candidats depuis: {args.candidates}")
        
        try:
            with open(args.references, 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f]
            
            with open(args.candidates, 'r', encoding='utf-8') as f:
                candidates = [line.strip() for line in f]
            
            if len(references) != len(candidates):
                logging.warning(
                    f"Le nombre de références ({len(references)}) ne correspond pas "
                    f"au nombre de candidats ({len(candidates)})"
                )
            
            data['references'] = references
            data['candidates'] = candidates
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données: {e}")
            sys.exit(1)
    else:
        # Utiliser les données d'exemple
        logging.info("Aucune donnée spécifiée, utilisation des données d'exemple")
        data['references'] = [
            "The cat is on the mat.",
            "The weather is nice today.",
            "I love reading books."
        ]
        data['candidates'] = [
            "There is a cat on the mat.",
            "We have good weather today.",
            "Books are my favorite thing to read."
        ]
    
    # Charger les scores humains si spécifiés
    if args.human_scores:
        logging.info(f"Chargement des scores humains depuis: {args.human_scores}")
        try:
            if args.human_scores.endswith('.csv'):
                df = pd.read_csv(args.human_scores)
                # Détecter la colonne de scores
                score_cols = [col for col in df.columns 
                             if any(term in col.lower() 
                                  for term in ['score', 'rating', 'human', 'eval', 'judgment'])]
                score_col = score_cols[0] if score_cols else df.columns[0]
                data['human_scores'] = df[score_col].tolist()
            elif args.human_scores.endswith('.json'):
                with open(args.human_scores, 'r') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    data['human_scores'] = json_data
                elif isinstance(json_data, dict) and 'scores' in json_data:
                    data['human_scores'] = json_data['scores']
            elif args.human_scores.endswith('.txt'):
                with open(args.human_scores, 'r') as f:
                    data['human_scores'] = [float(line.strip()) for line in f]
        except Exception as e:
            logging.error(f"Erreur lors du chargement des scores humains: {e}")
            sys.exit(1)
    
    return data


def main() -> None:
    """
    Point d'entrée principal du framework en ligne de commande.
    """
    # Parser les arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Configurer le logging
    setup_logging(args.verbose)
    
    # Traiter les fonctionnalités spéciales
    if args.create_config:
        config = create_default_config()
        save_config(config, args.create_config)
        logging.info(f"Configuration d'exemple créée et sauvegardée dans: {args.create_config}")
        return
    
    if args.list_metrics:
        list_available_metrics()
        return
    
    # Initialiser l'évaluateur
    evaluator = initialize_evaluator(args)
    
    # Vérifier qu'au moins une métrique est disponible
    if not evaluator.metrics:
        logging.error("Aucune métrique disponible pour l'évaluation.")
        sys.exit(1)
    
    # Charger les données
    data = load_data(args)
    
    # Créer les répertoires de sortie
    os.makedirs(args.output, exist_ok=True)
    visualizations_dir = os.path.join(args.output, "visualizations")
    if args.visualize:
        os.makedirs(visualizations_dir, exist_ok=True)
    
    # Évaluer et calculer les corrélations si des scores humains sont disponibles
    if 'human_scores' in data:
        logging.info(f"Évaluation avec corrélation aux scores humains ({len(data['human_scores'])} scores)")
        results = evaluator.evaluate_with_human_correlation(
            data['references'], 
            data['candidates'],
            data['human_scores'],
            correlation_methods=args.correlation_methods
        )
        
        # Ajouter les scores humains aux résultats pour la visualisation
        results['human_scores'] = data['human_scores']
        
        # Créer des visualisations si demandé
        if args.visualize:
            logging.info("Création des visualisations...")
            viz_files = create_correlation_dashboard(results, visualizations_dir)
            logging.info(f"{len(viz_files)} visualisations créées dans: {visualizations_dir}")
        
        # Afficher le rapport de corrélation
        print("\nRapport de corrélation avec les évaluations humaines:")
        print(results['correlation_report'])
        
        # Identifier la meilleure métrique
        best_metric, best_correlation = results['best_metric']
        print(f"\nLa métrique la plus corrélée avec les jugements humains: {best_metric} ({best_correlation:.3f})")
    else:
        logging.info("Évaluation sans corrélation (pas de scores humains disponibles)")
        results = evaluator.evaluate(data['references'], data['candidates'])
    
    # Afficher les résultats de l'évaluation
    print("\nRésultats de l'évaluation:")
    for metric_name, metric_results in results.get('evaluation_results', results).items():
        print(f"\n{metric_name}:")
        print(f"  Score global: {metric_results['score']:.4f}")
    
    # Générer et sauvegarder le rapport
    report = evaluator.generate_report(results, args.output)
    
    print("\nRésumé du rapport:")
    print(report.summary())
    logging.info(f"Rapport complet sauvegardé dans: {args.output}")


if __name__ == "__main__":
    main()
