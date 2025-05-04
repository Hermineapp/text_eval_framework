"""
Script to evaluate the SummEval dataset with multiple metrics.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import logging

# Configure logging to silence unwanted messages
logging.basicConfig(level=logging.ERROR)  # Only show ERROR level logs
logging.getLogger("httpx").setLevel(logging.ERROR)  # Silence httpx logs specifically

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the needed modules
from datasets import load_dataset
from core.evaluator import TextEvaluator
from metrics.bleu import BLEUMetric
from core.report import Report
from utils.visualization import create_correlation_dashboard

def evaluate_with_rouge(evaluator, machine_summaries, human_references, source_documents=None):
    """
    Correctly evaluate machine summaries against human reference summaries
    
    Args:
        evaluator: TextEvaluator instance
        machine_summaries: List of machine-generated summaries
        human_references: List of human reference summaries or list of lists for multiple references
        source_documents: Optional source documents (not used for ROUGE)
    
    Returns:
        Dict of ROUGE scores
    """
    # Handle single or multiple reference summaries
    if isinstance(human_references[0], list):
        # Multiple references - evaluate against each and take max score
        all_scores = []
        for i, machine_summary in enumerate(machine_summaries):
            references = human_references[i]
            # Create pairs of (machine_summary, reference) for each reference
            best_score = None
            best_result = None
            for ref in references:
                result = evaluator.evaluate([ref], [machine_summary])
                if 'rouge' in result and (best_score is None or result['rouge']['score'] > best_score):
                    best_score = result['rouge']['score']
                    best_result = result
            all_scores.append(best_result if best_result else {})
        return all_scores
    else:
        # Single reference - straightforward evaluation
        return evaluator.evaluate(human_references, machine_summaries)
    
def initialize_evaluator(metrics_to_use=None, verbose=True, model="qwen2.5:72b"):
    """
    Initialize the evaluator with the selected metrics.
    
    Args:
        metrics_to_use: List of metric names to use. If None, try to load all metrics.
        verbose: Whether to print detailed loading information.
        
    Returns:
        TextEvaluator: Initialized evaluator with metrics
    """
    evaluator = TextEvaluator()
    metrics_success = []
    metrics_failed = []
    
    # Dictionary of available metrics and their initialization functions
    all_metrics = {
        'bleu': lambda: evaluator.add_metric(BLEUMetric(weights=(0.25, 0.25, 0.25, 0.25), smoothing="method1")),
        
        'rouge': lambda: evaluator.add_metric(
            __import__('metrics.rouge', fromlist=['ROUGEMetric']).ROUGEMetric(
                rouge_types=['rouge1', 'rouge2', 'rougeL']
            )
        ),
        
        'bert_score': lambda: evaluator.add_metric(
            __import__('metrics.bert_score', fromlist=['BERTScoreMetric']).BERTScoreMetric(
                model_name="bert-base-uncased", batch_size=32
            )
        ),
        
        'meteor': lambda: evaluator.add_metric(
            __import__('metrics.meteor', fromlist=['METEORMetric']).METEORMetric(
                language='en', use_synonyms=True
            )
        ),
        'moverscore': lambda: evaluator.add_metric(
            __import__('metrics.moverscore', fromlist=['MoverScoreMetric']).MoverScoreMetric(

            )
        ),
        
        'sbert': lambda: evaluator.add_metric(
            __import__('metrics.sentence_bert', fromlist=['SentenceBERTMetric']).SentenceBERTMetric(
                model_name="all-MiniLM-L6-v2", similarity_metric="cosine"
            )
        ),
        
        'jaccard': lambda: evaluator.add_metric(
            __import__('example_custom', fromlist=['JaccardSimilarityMetric']).JaccardSimilarityMetric(
                tokenize=True, lowercase=True
            )
        ),
        
        'questeval': lambda: evaluator.add_metric(
            __import__('metrics.questeval', fromlist=['QuestEvalMetric']).QuestEvalMetric(
                task="summarization", language="en", no_cuda=True
            )
        ),
        
        'bartscore': lambda: evaluator.add_metric(
            __import__('metrics.bartscore', fromlist=['BartScoreMetric']).BartScoreMetric(
                model_name="facebook/bart-large-cnn", direction="src2tgt"
            )
        ),
        
        'unieval': lambda: evaluator.add_metric(
            __import__('metrics.unieval', fromlist=['UniEvalMetric']).UniEvalMetric(
                task="summarization", aspects=["coherence", "consistency"], verbose=False
            )
        ),
        
        'geval_rel': lambda: evaluator.add_metric(
            __import__('metrics.geval', fromlist=['GEvalMetric']).GEvalMetric(
                dimension="relevance",
                model_type="ollama",
                model_name=model,
                ollama_base_url="http://localhost:11434",
                n_responses=1,
                verbose=False,  # Disable verbose logging
                debug_mode=False  # Disable debug mode
            )
        ),
        'geval_con': lambda: evaluator.add_metric(
            __import__('metrics.geval', fromlist=['GEvalMetric']).GEvalMetric(
                dimension="consistency",
                model_type="ollama",
                model_name=model,
                ollama_base_url="http://localhost:11434",
                n_responses=1,
                verbose=False,  # Disable verbose logging
                debug_mode=False  # Disable debug mode
            )
        ),
        'geval_coh': lambda: evaluator.add_metric(
            __import__('metrics.geval', fromlist=['GEvalMetric']).GEvalMetric(
                dimension="coherence",
                model_type="ollama",
                model_name=model,
                ollama_base_url="http://localhost:11434",
                n_responses=1,
                verbose=False,  # Disable verbose logging
                debug_mode=False  # Disable debug mode
            )
        ),
        'geval_flu': lambda: evaluator.add_metric(
            __import__('metrics.geval', fromlist=['GEvalMetric']).GEvalMetric(
                dimension="fluency",
                model_type="ollama",
                model_name=model,
                ollama_base_url="http://localhost:11434",
                n_responses=1,
                verbose=False,  # Disable verbose logging
                debug_mode=False  # Disable debug mode
            )
        ),
        'seval_ex': lambda: evaluator.add_metric(
            __import__('metrics.seval_ex', fromlist=['SEvalExMetric']).SEvalExMetric(
                model_name=model,
                ollama_base_url="http://localhost:11434",
                max_retries=2,
                verbose=False,  # Disable verbose logging
                debug_mode=False  # Disable debug mode
            )
        )
    }
    """'seval_ex': lambda: evaluator.add_metric(
        __import__('metrics.seval_ex', fromlist=['SEvalExMetric']).SEvalExMetric(
            model_name=model,
            ollama_base_url="http://localhost:11434",
            max_retries=2,
            verbose=False,  # Disable verbose logging
            debug_mode=False  # Disable debug mode
        )
    )"""

    all_metrics = {

        'rougeHF': lambda: evaluator.add_metric(
            __import__('metrics.HFrouge', fromlist=['HuggingFaceRougeMetric']).HuggingFaceRougeMetric(
                rouge_types=['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True,
            )
        ),

    }


    # If specific metrics requested, filter to just those
    if metrics_to_use:
        metric_funcs = {k: v for k, v in all_metrics.items() if k in metrics_to_use}
    else:
        metric_funcs = all_metrics
    
    # Try to load each metric
    for metric_name, metric_func in metric_funcs.items():
        if verbose:
            print(f"Loading metric: {metric_name}...")
        
        try:
            metric_func()
            metrics_success.append(metric_name)
            if verbose:
                print(f"✓ Metric {metric_name} loaded successfully")
        except Exception as e:
            metrics_failed.append((metric_name, str(e)))
            if verbose:
                print(f"✗ Failed to load {metric_name}: {e}")
    
    if verbose:
        print(f"\n✅ {len(metrics_success)} metrics loaded: {', '.join(metrics_success)}")
        if metrics_failed:
            print(f"❌ {len(metrics_failed)} metrics failed to load:")
            for name, error in metrics_failed:
                print(f"  - {name}: {error}")
    
    return evaluator


def calculate_correlations(results):
    """
    Calculate correlations between metric scores and human judgments.

    Args:
        results: List of evaluation results

    Returns:
        dict: Correlation results
    """
    from scipy.stats import pearsonr, spearmanr

    # Extract metric names
    metric_names = set()
    for r in results:
        metric_names.update(r["metric_scores"].keys())

    # Human judgment dimensions
    dimensions = ["coherence", "consistency", "fluency", "relevance"]

    # Calculate correlations
    correlations = {
        "pearson": {dim: {} for dim in dimensions},
        "spearman": {dim: {} for dim in dimensions}
    }

    for dim in dimensions:
        for metric in metric_names:
            # Get pairs of human and metric scores
            pairs = [(r["human_scores"][dim], r["metric_scores"].get(metric, float('nan')))
                     for r in results if metric in r["metric_scores"]]

            if not pairs:
                continue

            human_scores, metric_scores = zip(*pairs)

            # Filter out any NaN values
            valid_pairs = [(h, m) for h, m in zip(human_scores, metric_scores)
                           if not (np.isnan(h) or np.isnan(m))]

            if len(valid_pairs) < 2:
                # Not enough data for correlation
                continue

            valid_human, valid_metric = zip(*valid_pairs)

            try:
                p_corr, p_p = pearsonr(valid_human, valid_metric)
                s_corr, s_p = spearmanr(valid_human, valid_metric)

                correlations["pearson"][dim][metric] = {
                    "correlation": p_corr,
                    "p_value": p_p,
                    "sample_size": len(valid_pairs)
                }

                correlations["spearman"][dim][metric] = {
                    "correlation": s_corr,
                    "p_value": s_p,
                    "sample_size": len(valid_pairs)
                }
            except Exception as e:
                print(f"Error calculating correlation for {metric} on {dim}: {e}")

    return correlations


def evaluate_summeval(metrics_to_use=None, max_examples=None, max_summaries_per_example=None,
                      output_dir="summeval_results", batch_size=1, save_interim_frequency=10, model_name="qwen2.5:72b"):
    """
    Evaluate the SummEval dataset using multiple metrics.

    Args:
        metrics_to_use: List of metric names to use
        max_examples: Maximum number of source documents to process
        max_summaries_per_example: Maximum summaries to evaluate per source document
        output_dir: Directory to save results
        batch_size: Number of examples to process at once
        save_interim_frequency: Save interim results after processing this many documents

    Returns:
        dict: Evaluation results
    """
    print("Loading SummEval dataset...")
    dataset = load_dataset("mteb/summeval")["test"]

    # Limit examples if specified
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Initialize the evaluator
    evaluator = initialize_evaluator(metrics_to_use, model=model_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process examples in batches
    all_results = []

    # Calculate total number of summaries to evaluate
    total_summaries = 0
    for example in dataset:
        if max_summaries_per_example:
            total_summaries += min(len(example["machine_summaries"]), max_summaries_per_example)
        else:
            total_summaries += len(example["machine_summaries"])

    print(f"Processing {len(dataset)} documents with {total_summaries} summaries in total...")

    # Initialize a single progress bar for all summaries
    progress_bar = tqdm(total=total_summaries, desc="Evaluating summaries")

    # Track the current timestamp to use for interim saves
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    interim_path = os.path.join(output_dir, f"interim_results_{timestamp}.json")

    summaries_processed = 0

    for doc_idx, example in enumerate(dataset):
        doc_id = doc_idx  # Use index as document ID
        source_text = example["text"]
        machine_summaries = example["machine_summaries"]

        # Limit summaries per example if specified
        if max_summaries_per_example:
            machine_summaries = machine_summaries[:max_summaries_per_example]

        # Process all summaries for this document
        doc_results = []

        # Process summaries in batches
        for i in range(0, len(machine_summaries), batch_size):
            batch_summaries = machine_summaries[i:i + batch_size]
            batch_human_scores = {
                "coherence": example["coherence"][i:i + batch_size],
                "consistency": example["consistency"][i:i + batch_size],
                "fluency": example["fluency"][i:i + batch_size],
                "relevance": example["relevance"][i:i + batch_size],
            }

            # Create lists of references and candidates for this batch
            references = [source_text] * len(batch_summaries)
        
            # Evaluate with all metrics
            # try:
            #     eval_results = evaluator.evaluate(references, batch_summaries)
            # except Exception as e:
            #     print(f"Error evaluating during evaluator summaries {i}:{i + batch_size} for document {doc_id}: {e}")
            #     continue

            try:
                # Get human reference summaries for this document
                if "human_summaries" in example:
                    human_refs = example["human_summaries"]
                    # Special handling for ROUGE - use human references instead of source text
                    rouge_metric = None
                    for metric_name, metric in evaluator.metrics.items():
                        if metric_name == 'rouge':
                            rouge_metric = metric
                            break
                            
                    if rouge_metric:
                        # Remove ROUGE from evaluator temporarily
                        evaluator.metrics.pop('rouge', None)
                        
                        # check if there is other metrics

                        # Evaluate other metrics normally (using source text as reference)
                        if len(evaluator.metrics) != 0:
                            eval_results = evaluator.evaluate(references, batch_summaries)
                        
                        # Evaluate ROUGE separately with human references
                        rouge_results = evaluate_with_rouge(
                            TextEvaluator([('rouge', rouge_metric)]), 
                            batch_summaries,
                            [human_refs] * len(batch_summaries)  # Provide same refs for each summary
                        )
                        
                        # Merge results
                        for i, rouge_result in enumerate(rouge_results):
                            if rouge_result and 'rouge' in rouge_result:
                                eval_results['rouge'] = rouge_result['rouge']
                        
                        # Restore ROUGE to evaluator
                        evaluator.metrics['rouge'] = rouge_metric
                    else:
                        # Just evaluate normally if no ROUGE metric
                        eval_results = evaluator.evaluate([source_text] * len(batch_summaries), batch_summaries)
                else:
                    # No human references available, evaluate normally
                    eval_results = evaluator.evaluate([source_text] * len(batch_summaries), batch_summaries)
            except Exception as e:
                print(f"Error evaluating summaries {i}:{i + batch_size} for document {doc_id}: {e}")
                continue


            try:
                # Store results for each summary in the batch
                for j, summary in enumerate(batch_summaries):
                    summary_results = {
                        "doc_id": doc_id,
                        "summary_id": i + j,
                        "source_text": source_text,
                        "summary": summary,
                        "human_scores": {
                            dim: batch_human_scores[dim][j]
                            for dim in batch_human_scores
                        },
                        "metric_scores": {}
                    }

                    # Extract individual scores for this summary
                    for metric_name, metric_result in eval_results.items():
                        if "individual_scores" in metric_result and len(metric_result["individual_scores"]) > j:
                            summary_results["metric_scores"][metric_name] = metric_result["individual_scores"][j]

                    # Special case for ROUGE - extract ROUGE-1, ROUGE-2, and ROUGE-L separately
                    # First check if "rouge" is in the eval_results
                    if "rouge" in eval_results:
                        rouge_results = eval_results["rouge"]
                        
                        # Add all three ROUGE metrics
                        if "rouge1" in rouge_results:
                            summary_results["metric_scores"]["rouge_1"] = rouge_results["rouge1"]["individual_scores"][j]
                        
                        if "rouge2" in rouge_results:
                            summary_results["metric_scores"]["rouge_2"] = rouge_results["rouge2"]["individual_scores"][j]
                        
                        if "rougeL" in rouge_results:
                            summary_results["metric_scores"]["rouge_L"] = rouge_results["rougeL"]["individual_scores"][j]

                        for key in ['1', '2', 'L']:
                            if key in eval_results["rouge"]:
                                rouge_key = f"rouge_{key}"
                                summary_results["metric_scores"][rouge_key] = eval_results["rouge"][key]["individual_scores"][j]


                    doc_results.append(summary_results)

            except Exception as e:
                print(f"Error processing document {doc_id}, summaries {i}:{i + batch_size}: {e}")
                if "rouge" in eval_results:
                    print("Rouge results structure:", eval_results["rouge"].keys())
                    if "1" in eval_results["rouge"]:
                        print("Rouge-1 structure:", eval_results["rouge"]["1"].keys())
                continue

            # Update progress for this batch
            progress_bar.update(len(batch_summaries))
            summaries_processed += len(batch_summaries)

        # Add the results for this document to the overall results
        all_results.extend(doc_results)

        # Save interim results periodically
        if (doc_idx + 1) % save_interim_frequency == 0:
            with open(interim_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(
                f"\nSaved interim results ({summaries_processed}/{total_summaries} summaries processed) to {interim_path}")

    progress_bar.close()

    # Calculate correlation with human judgments
    correlation_results = calculate_correlations(all_results)

    # Save final results
    results_path = os.path.join(output_dir, "summeval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            f"summary_evaluations_{model_name}": all_results,
            f"correlations_{model_name}": correlation_results
        }, f, indent=2)

    # Also save as CSV for easier analysis
    summaries_df = pd.DataFrame([{
        "doc_id": r["doc_id"],
        "summary_id": r["summary_id"],
        "summary": r["summary"],
        "coherence": r["human_scores"]["coherence"],
        "consistency": r["human_scores"]["consistency"],
        "fluency": r["human_scores"]["fluency"],
        "relevance": r["human_scores"]["relevance"],
        **{f"metric_{k}": v for k, v in r["metric_scores"].items()}
    } for r in all_results])

    csv_path = os.path.join(output_dir, f"summeval_results_{model_name}.csv")
    summaries_df.to_csv(csv_path, index=False)

    # Generate correlation heatmap
    plot_path = os.path.join(output_dir, f"correlation_heatmap_{model_name}.png")
    plot_correlation_heatmap(correlation_results, plot_path)

    print(f"Evaluation complete! Results saved to {output_dir}")

    # Add timing report
    print("\nRapport de performance des métriques:")
    evaluator.print_timing_report()

    return {"summary_evaluations": all_results, "correlations": correlation_results}



def plot_correlation_heatmap(correlation_results, output_path):
    """
    Create a heatmap of correlations between metrics and human judgments.
    
    Args:
        correlation_results: Correlation results dictionary
        output_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract Spearman correlations
        dimensions = list(correlation_results["spearman"].keys())
        metrics = set()
        for dim_results in correlation_results["spearman"].values():
            metrics.update(dim_results.keys())
        
        metrics = sorted(metrics)
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(metrics), len(dimensions)))
        
        for i, metric in enumerate(metrics):
            for j, dim in enumerate(dimensions):
                if metric in correlation_results["spearman"][dim]:
                    corr_matrix[i, j] = correlation_results["spearman"][dim][metric]["correlation"]
                else:
                    corr_matrix[i, j] = np.nan
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                   xticklabels=dimensions, yticklabels=metrics)
        plt.title('Spearman Correlation between Metrics and Human Judgments')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {output_path}")
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SummEval dataset with multiple metrics")
    parser.add_argument("--metrics", nargs="+", help="Metrics to use (space-separated list)")
    parser.add_argument("--max_examples", type=int, default=100,
                       help="Maximum number of documents to process")
    parser.add_argument("--max_summaries", type=int, default=16,
                       help="Maximum summaries per document")
    parser.add_argument("--output_dir", type=str, default="summeval_results",
                       help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="gemma3:27b",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Number of summaries to process at once")
    parser.add_argument("--save_frequency", type=int, default=10,
                       help="Save interim results after processing this many documents")
    
    args = parser.parse_args()
    
    evaluate_summeval(
        metrics_to_use=args.metrics,
        max_examples=args.max_examples,
        max_summaries_per_example=args.max_summaries,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_interim_frequency=args.save_frequency,
        model_name=args.model_name,
    )

