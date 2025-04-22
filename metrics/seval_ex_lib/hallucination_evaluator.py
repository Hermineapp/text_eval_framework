import argparse
import json
from typing import Dict, List, Optional, Union
from enum import Enum
import ollama
from datasets import load_dataset
from tqdm import tqdm
import random

class HallucinationType(Enum):
    """Enum for different types of hallucinations that can be generated."""
    ENTITY_REPLACEMENT = "entity_replacement"
    INCORRECT_EVENT = "incorrect_event"
    FICTITIOUS_DETAILS = "fictitious_details"

class HallucinationEvaluator:
    """
        Pipeline for generating and evaluating hallucinated summaries.

        Implements the hallucination detection methodology described in Section 4.2
        of the paper, which tests SEval-Ex's robustness against different types of
        hallucinations:
        1. Entity Replacement: Replacing named entities with incorrect ones
        2. Incorrect Events: Modifying sequence of events
        3. Fictitious Details: Adding plausible but unsupported details
        """

    def __init__(self, model_name: str = "qwen2.5:72b", temperature: float = 0.0001):
        """
        Initialize the hallucination evaluator.

        Args:
            model_name (str): Name of the LLM model to use for hallucination generation
            temperature (float): Temperature parameter for controlling randomness in generation
                Lower values (near 0) produce more deterministic outputs
        """

        self.model_name = model_name
        self.temperature = temperature
        self.prompts = {
            HallucinationType.ENTITY_REPLACEMENT: """
            Replace named entities (people, organizations, locations, dates) with incorrect but plausible alternatives in the following summary.
            Keep the same structure and length, only replace the entities.
            Do not add any explanations, only provide the modified summary.
            
            Summary: {summary}
            
            Modified summary:""",
            
            HallucinationType.INCORRECT_EVENT: """
            Modify the sequence of events and their relationships in the following summary to make them factually incorrect.
            Keep similar length and style, but change the order, causation, or outcome of events.
            Do not add any explanations, only provide the modified summary.
            
            Summary: {summary}
            
            Modified summary:""",
            
            HallucinationType.FICTITIOUS_DETAILS: """
            Add plausible but completely fictional details to the following summary.
            Keep the main points but embellish them with made-up specifics, numbers, or contextual information.
            Do not add any explanations, only provide the modified summary.
            
            Summary: {summary}
            
            Modified summary:"""
        }

    def generate_hallucination(self, summary: str, hallucination_type: HallucinationType) -> str:
        """
        Generate a hallucinated version of the summary.

        Creates controlled hallucinations according to the specified type, while
        maintaining the overall structure and style of the original summary.

        Args:
            summary (str): Original summary text
            hallucination_type (HallucinationType): Type of hallucination to generate

        Returns:
            str: Hallucinated version of the summary
        """
        prompt = self.prompts[hallucination_type].format(summary=summary)
        
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": self.temperature}
        )
        
        return response['response'].strip()

    def evaluate_with_seval(self, 
                          hallucinated_summary: str, 
                          source_text: str, 
                          seval_pipeline) -> Dict:
        """
            Evaluate hallucinated summary using SEval pipeline.

            Args:
                hallucinated_summary (str): Generated hallucinated summary
                source_text (str): Original source text
                seval_pipeline: Instance of SEvalPipeline for evaluation

            Returns:
                Dict: Evaluation metrics
        """
        return seval_pipeline.evaluate_single(source_text, hallucinated_summary)

    def evaluate_summeval_dataset(self, 
                                dataset_name: str = "mteb/summeval",
                                seval_pipeline = None,
                                sample_size: Optional[int] = None,
                                split: str = "test") -> Dict[str, List[Dict]]:
        """
        Evaluate hallucinations on the SummEval dataset.

        Creates and evaluates hallucinated versions of summaries from the SummEval dataset,
        dividing the dataset into equal parts for each hallucination type.

        Args:
            dataset_name (str): Name of the dataset (default: "mteb/summeval")
            seval_pipeline: SEval pipeline instance for evaluation
            sample_size (Optional[int]): Number of samples to evaluate
            split (str): Dataset split to use

        Returns:
            Dict[str, List[Dict]]: Results organized by hallucination type
        """

        # Load dataset
        dataset = load_dataset(dataset_name)[split]
        
        if sample_size:
            # Randomly sample entries while maintaining even distribution
            indices = list(range(len(dataset)))
            random.seed(42)  # For reproducibility
            sample_indices = random.sample(indices, sample_size)
            dataset = [dataset[i] for i in sample_indices]
        
        results = {h_type.value: [] for h_type in HallucinationType}
        
        # Calculate chunk size for different hallucination types
        chunk_size = len(dataset) // len(HallucinationType)
        
        for i, entry in enumerate(tqdm(dataset, desc="Evaluating summaries")):
            # Determine hallucination type based on position
            h_type = list(HallucinationType)[i // chunk_size]
            
            # Process each machine summary in the entry
            for j, summary in enumerate(entry['machine_summaries']):
                # Generate hallucinated version
                hallucinated_summary = self.generate_hallucination(summary, h_type)
                
                # Evaluate using SEval if provided
                evaluation = None
                if seval_pipeline:
                    evaluation = self.evaluate_with_seval(
                        hallucinated_summary,
                        entry['text'],
                        seval_pipeline
                    )
                
                # Store results
                results[h_type.value].append({
                    "text": entry['text'],
                    "original_summary": summary,
                    "hallucinated_summary": hallucinated_summary,
                    "original_metrics": {
                        "consistency": entry['consistency'][j],
                        "relevance": entry['relevance'][j],
                        "coherence": entry['coherence'][j],
                        "fluency": entry['fluency'][j]
                    },
                    "seval_evaluation": evaluation
                })
                
        return results

    def evaluate_single(self,
                       text: str,
                       summary: str,
                       seval_pipeline = None) -> Dict[str, Dict]:
        """
        Evaluate hallucinations for a single text-summary pair.

        Generates hallucinated versions of the summary for each hallucination type
        and evaluates them using the SEval pipeline.

        Args:
            text (str): Source text
            summary (str): Original summary
            seval_pipeline: Optional SEval pipeline instance

        Returns:
            Dict[str, Dict]: Results for each hallucination type
        """

        results = {}
        
        for h_type in HallucinationType:
            # Generate hallucinated version
            hallucinated = self.generate_hallucination(summary, h_type)
            
            # Evaluate using SEval if provided
            evaluation = None
            if seval_pipeline:
                evaluation = self.evaluate_with_seval(hallucinated, text, seval_pipeline)
            
            results[h_type.value] = {
                "original_summary": summary,
                "hallucinated_summary": hallucinated,
                "seval_evaluation": evaluation
            }
            
        return results

    def analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze evaluation results across different hallucination types.

        Calculates aggregated metrics for each hallucination type to measure
        the impact of different types of hallucinations on evaluation performance.

        Args:
            results (Dict[str, List[Dict]]): Evaluation results

        Returns:
            Dict[str, Dict[str, float]]: Aggregated metrics by hallucination type
        """
        analysis = {}
        
        for h_type, entries in results.items():
            # Calculate average metrics
            metrics = {
                "avg_original_consistency": sum(e["original_metrics"]["consistency"] 
                                             for e in entries) / len(entries),
            }
            
            # Add SEval metrics if available
            if entries[0].get("seval_evaluation"):
                metrics.update({
                    "avg_seval_precision": sum(e["seval_evaluation"]["metrics"]["precision"] 
                                             for e in entries) / len(entries),
                    "avg_seval_tp_count": sum(e["seval_evaluation"]["metrics"]["tp_count"] 
                                            for e in entries) / len(entries),
                    "avg_seval_fp_count": sum(e["seval_evaluation"]["metrics"]["fp_count"] 
                                            for e in entries) / len(entries)
                })
            
            analysis[h_type] = metrics
            
        return analysis

def main():
    """
        Main function to parse command-line arguments and run evaluation.

        Provides an interface for:
        1. Evaluating a single text-summary pair
        2. Evaluating an entire dataset
        3. Generating and evaluating hallucinated summaries

        Command-line arguments:
            --model: Model name to use
            --method: Evaluation method
            --chunk-size: Number of sentences per chunk
            --max-text-length: Maximum text length before chunking
            --dataset: Dataset to evaluate
            --text: Source text for single evaluation
            --summary: Summary text for single evaluation
            --output: Output file path
    """
    parser = argparse.ArgumentParser(description="Hallucination Evaluation Pipeline")
    parser.add_argument("--model", default="qwen2.5:72b", help="Model name to use")
    parser.add_argument("--sample-size", type=int, help="Number of samples to evaluate")
    parser.add_argument("--output", default="hallucination_results.json", 
                       help="Output file path")
    parser.add_argument("--text", help="Source text for single evaluation")
    parser.add_argument("--summary", help="Summary text for single evaluation")
    parser.add_argument("--dataset", action="store_true", 
                       help="Use SummEval dataset instead of single text")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HallucinationEvaluator(model_name=args.model)
    
    try:
        from seval_package import SEvalPipeline
        seval = SEvalPipeline()
    except ImportError:
        print("SEval package not found. Running without SEval evaluation.")
        seval = None
    
    if args.dataset:
        # Dataset evaluation mode
        print("Evaluating SummEval dataset...")
        results = evaluator.evaluate_summeval_dataset(
            sample_size=args.sample_size,
            seval_pipeline=seval
        )
        analysis = evaluator.analyze_results(results)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump({
                "results": results,
                "analysis": analysis
            }, f, indent=2)
            
        # Print analysis
        print("\nAnalysis by Hallucination Type:")
        for h_type, metrics in analysis.items():
            print(f"\n{h_type}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
                
    elif args.text and args.summary:
        # Single text evaluation mode
        print("Evaluating single text/summary pair...")
        results = evaluator.evaluate_single(
            args.text,
            args.summary,
            seval_pipeline=seval
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print results
        print("\nGenerated Hallucinations:")
        for h_type, result in results.items():
            print(f"\n{h_type}:")
            print(f"Original: {result['original_summary']}")
            print(f"Hallucinated: {result['hallucinated_summary']}")
            if result.get('seval_evaluation'):
                print("SEval Metrics:")
                print(json.dumps(result['seval_evaluation']['metrics'], indent=2))
                
    else:
        print("Please provide either --dataset flag or both --text and --summary arguments")

if __name__ == "__main__":
    main()
