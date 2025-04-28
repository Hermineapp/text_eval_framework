"""
Exemple complet d'utilisation du framework avec toutes les métriques disponibles.
"""

import sys
import os
import argparse
import pandas as pd

# Ajouter le répertoire parent au chemin de recherche
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import des métriques et de l'évaluateur
from core.evaluator import TextEvaluator
from metrics.bleu import BLEUMetric
from core.report import Report
from utils.visualization import create_correlation_dashboard
from example_custom import JaccardSimilarityMetric

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
    
    print(f"Données d'exemple créées et sauvegardées dans: {csv_path}")
    return data


def main():
    # Créer l'évaluateur
    evaluator = TextEvaluator()
    
    # 1. Ajouter BLEU (métrique de base)
    try:
        print("Ajout de la métrique BLEU...")
        evaluator.add_metric(BLEUMetric(weights=(0.25, 0.25, 0.25, 0.25), smoothing="method1"))
        print("✓ Métrique BLEU ajoutée avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique BLEU: {e}")
    
    # 2. Ajouter ROUGE (nécessite rouge_score)
    try:
        print("\nAjout de la métrique ROUGE...")
        from metrics.rouge import ROUGEMetric
        evaluator.add_metric(ROUGEMetric(rouge_types=['rouge1', 'rouge2', 'rougeL']))
        print("✓ Métrique ROUGE ajoutée avec succès")
    except ImportError:
        print("✗ Impossible d'ajouter la métrique ROUGE: package 'rouge_score' non installé")
        print("  Installez-le avec: pip install rouge_score")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique ROUGE: {e}")
    
    # 3. Ajouter BERTScore (nécessite bert-score et torch)
    try:
        print("\nAjout de la métrique BERTScore...")
        from metrics.bert_score import BERTScoreMetric
        evaluator.add_metric(BERTScoreMetric(model_name="bert-base-uncased", batch_size=32))
        print("✓ Métrique BERTScore ajoutée avec succès")
    except ImportError:
        print("✗ Impossible d'ajouter la métrique BERTScore: packages 'bert-score' et/ou 'torch' non installés")
        print("  Installez-les avec: pip install torch bert-score")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique BERTScore: {e}")
    
    # 4. Ajouter METEOR
    try:
        print("\nAjout de la métrique METEOR...")
        from metrics.meteor import METEORMetric
        evaluator.add_metric(METEORMetric(language='en', use_synonyms=True))
        print("✓ Métrique METEOR ajoutée avec succès")
    except ImportError:
        print("✗ Impossible d'ajouter la métrique METEOR: composant NLTK manquant")
        print("  Exécutez: python -m nltk.downloader wordnet punkt")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique METEOR: {e}")
    
    # 5. Ajouter SentenceBERT (nécessite sentence-transformers)
    try:
        print("\nAjout de la métrique SentenceBERT...")
        from metrics.sentence_bert import SentenceBERTMetric
        evaluator.add_metric(SentenceBERTMetric(model_name="all-MiniLM-L6-v2", similarity_metric="cosine"))
        print("✓ Métrique SentenceBERT ajoutée avec succès")
    except ImportError:
        print("✗ Impossible d'ajouter la métrique SentenceBERT: package 'sentence-transformers' non installé")
        print("  Installez-le avec: pip install sentence-transformers")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique SentenceBERT: {e}")
    
    # 6. Ajouter la métrique personnalisée Jaccard
    try:
        print("\nAjout de la métrique Jaccard (personnalisée)...")
        evaluator.add_metric(JaccardSimilarityMetric(tokenize=True, lowercase=True))
        print("✓ Métrique Jaccard ajoutée avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique Jaccard: {e}")
    

    # 7. Ajouter QuestEval (nécessite questeval)
    try:
        print("\nAjout de la métrique QuestEval...")
        from metrics.questeval import QuestEvalMetric
        evaluator.add_metric(QuestEvalMetric(task="summarization", language="en", no_cuda=False))
        print("✓ Métrique QuestEval ajoutée avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique QuestEval: {e}")

    # 8. Ajouter BartScore (nécessite transformers et torch)
    try:
        print("\nAjout de la métrique BartScore...")
        from metrics.bartscore import BartScoreMetric
        evaluator.add_metric(BartScoreMetric(model_name="facebook/bart-large-cnn", direction="avg"))
        print("✓ Métrique BartScore ajoutée avec succès")
    except ImportError:
        print("✗ Impossible d'ajouter la métrique BartScore: packages 'transformers' et/ou 'torch' non installés")
        print("  Installez-les avec: pip install torch transformers")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique BartScore: {e}")
    
    # 9. Ajouter UniEval (nécessite torch, nltk et unieval)
    try:
        print("\nAjout de la métrique UniEval...")
        from metrics.unieval import UniEvalMetric
        print("UniEval est disponible")
        evaluator.add_metric(UniEvalMetric(task="summarization", aspects=["coherence", "consistency"]))
        print("✓ Métrique UniEval ajoutée avec succès")
    #except ImportError:
    #    print("✗ Impossible d'ajouter la métrique UniEval: packages 'torch', 'nltk' ou 'unieval' non installés")
    #    print("  Installez-les avec: pip install torch nltk git+https://github.com/maszhongming/UniEval.git")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique UniEval: {e}")

    # 10. Ajouter la métrique G-Eval
    try:
        
        print(f"\nAjout de la métrique G-Eval")
        from metrics.geval import GEvalMetric
        evaluator.add_metric(
            GEvalMetric(
                dimension= "relevance",
                model_type="ollama",
                model_name="gemma3:27b",
                ollama_base_url="http://localhost:32149",
                debug_mode=True,
            )
        )
        print("✓ Métrique G-Eval ajoutée avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique G-Eval: {e}")


    # 11. Ajouter la métrique SEval_ex
    try:
        print(f"\nAjout de la métrique SEval_ex")
        from metrics.seval_ex import SEvalExMetric
        evaluator.add_metric(
            SEvalExMetric(
                model_name="gemma3:27b",
                ollama_base_url="http://localhost:32149",
                debug_mode=True,
            )
        )
        print("✓ Métrique SEval_ex ajoutée avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'ajout de la métrique SEval_ex: {e}")

    # Vérifier qu'au moins une métrique a été ajoutée
    if not evaluator.metrics:
        print("\n❌ Aucune métrique n'a pu être ajoutée. Vérifiez les dépendances et les erreurs ci-dessus.")
        return
    
    print(f"\n✅ {len(evaluator.metrics)} métriques ajoutées au total.")
    print("Métriques disponibles:", ", ".join(evaluator.metrics.keys()))
    
    # Créer les données d'exemple
    print("\nCréation des données d'exemple...")
    data_dir = os.path.join(parent_dir, "data")
    data = create_example_data(data_dir)
    
    # Préparer les répertoires de sortie
    output_dir = os.path.join(parent_dir, "results_all_metrics")
    os.makedirs(output_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Évaluer avec corrélation aux scores humains
    print("\nÉvaluation avec toutes les métriques et corrélation aux scores humains...")
    results = evaluator.evaluate_with_human_correlation(
        data['reference'],
        data['candidate'],
        data['human_score'],
        correlation_methods=['pearson', 'spearman', 'kendall']
    )
    
    # Ajouter les scores humains aux résultats pour la visualisation
    results['human_scores'] = data['human_score']
    
    # Créer des visualisations
    print("Création des visualisations...")
    viz_files = create_correlation_dashboard(results, visualizations_dir)
    print(f"{len(viz_files)} visualisations créées dans: {visualizations_dir}")
    
    # Afficher le rapport de corrélation
    print("\nRapport de corrélation avec les évaluations humaines:")
    print(results['correlation_report'])
    
    # Identifier la meilleure métrique
    best_metric, best_correlation = results['best_metric']
    print(f"\nLa métrique la plus corrélée avec les jugements humains: {best_metric} ({best_correlation:.3f})")
    
    # Générer et sauvegarder le rapport
    report = Report(results)
    saved_files = report.save(output_dir, prefix="all_metrics_report")
    
    print("\nRésumé du rapport:")
    print(report.summary())
    print(f"\nRapport complet sauvegardé dans: {output_dir}")
    print("Fichiers générés:", ", ".join(saved_files.values()))


if __name__ == "__main__":
    main()
