import pickle
import random
import ollama
import requests
import re
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from termcolor import colored
from pyfiglet import Figlet
import csv
from collections import defaultdict
import ollama
import requests
import re
import json
from typing import List, Dict, Tuple

# Ollama API endpoint
#OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Model hyperparameters
MODEL_NAME = "qwen2.5:72b"  #"llama3.1:70b"  # or "mistral" or "gemma"
MAX_TOKENS = 1000
TEMPERATURE = 0.0001

i = 0


# Parsing strategy
PARSING_STRATEGY = "regex"  # or "constrained"

# Prompts
STATEMENT_EXTRACTION_PROMPT = """
Extract key statements from the following text. Each statement should be a single, self-contained fact or claim. Do not add any ponctuation or words. Each statement should be on a new line.

Here an example of a statement extraction: 
Text: 
"Albert Einstein was born in Barcelona Spain, 1879" 
Extracted statements:
"Albert Einstein was born in Spain",
"Albert Einstein was born in Barcelona",
"Albert Einstein was born in 1879"


Text: {text}

Extracted statements:
"""

CORRECTNESS_LABELING_PROMPT = """
Compare the following statements from the summary with the statements from the original text. Do not add any ponctuation or words. Do not justify your answer.
Label each summary statement as:
- TP (True Positive): If the statement appears in the summary and is directly supported by a statement from the original text.
- FP (False Positive): If the statement appears in the summary but is not directly supported by a statement from the original text.
- FN (False Negative): If it appears in the original text but does not support any statement from the summary.

As you can see in the example bellow, first you have to concatenate the summary statements and the original text statements. 
Then you have to label each statement as TP, FP or FN.

Example: 
Summary Statements:
"Albert Einstein was born in Germany",
"Albert Einstein was born in 1879"

Original Text Statements:
"Albert Einstein was born in Spain",
"Albert Einstein was born in Barcelona",
"Albert Einstein was born in 1879"

Labels:
Albert Einstein was born in Spain. VERDICT: FP
Albert Einstein was born in Barcelona. VERDICT: FP
Albert Einstein was born in 1879. VERDICT: TP
Albert Einstein was born in Germany. VERDICT: FN

END Example



Summary Statements:
{summary_statements}

Original Text Statements:
{original_statements}

Labels:
"""

FAITHFULNESS_LABELING_PROMPT = """
Compare the following statements from the summary with the original text.
Label each statement as:
- PASSED: If the statement can be inferred from the original text.
- FAILED: If the statement cannot be inferred from the original text.

Summary Statements:
{summary_statements}

Original Text:
{original_text}

Labels:
"""

# use RELATION data
RELATION = False

# Regular expressions for parsing
REGEX_PATTERNS = {
    "correctness": {
        "TP": r"\bVERDICT:.*TP\b",
        "FP": r"\bVERDICT:.*FP\b",
        "FN": r"\bVERDICT:.*FN\b"
    },
    "faithfulness": {
        "PASSED": r"\bVERDICT:.*PASSED\b",
        "FAILED": r"\bVERDICT:.*FAILED\b"
    }
}

# JSON schemas for constrained generation
JSON_SCHEMAS = {
    "correctness": {
        "type": "object",
        "properties": {
            "TP": {"type": "array", "items": {"type": "string"}},
            "FP": {"type": "array", "items": {"type": "string"}},
            "FN": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["TP", "FP", "FN"]
    },
    "faithfulness": {
        "type": "object",
        "properties": {
            "PASSED": {"type": "array", "items": {"type": "string"}},
            "FAILED": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["PASSED", "FAILED"]
    }
}

CSV_FILE = "./save_csv/evaluation_results_chunk_3sentences_text_all_dataset.csv"
PKL_FILE_SAVE = "./save/St_chunk_3.pkl"


# Function to generate 10 random indices
def generate_random_indices(data_length, size=10):
    return [random.randint(0, data_length - 1) for _ in range(size)]


def ascii_bar(value, max_width=50):
    bar_width = int(value * max_width)
    return f"[{'#' * bar_width}{'-' * (max_width - bar_width)}] {value:.4f}"


def visualize_results(results: Dict[str, float], metric: str):
    print_step(5, f"Visualizing {metric.capitalize()} Results")
    max_key_length = max(len(key) for key in results.keys())
    for key, value in results.items():
        print(f"{key.rjust(max_key_length)}: {ascii_bar(value)}")
    print('index: ', i)


def print_step(step_number, description):
    f = Figlet(font='slant')
    print('\n' + '=' * 80)
    print(colored(f.renderText(f'Step {step_number}'), 'cyan'))
    print(colored(description, 'yellow'))
    print('=' * 80 + '\n')
    print('index: ', i)


def print_data_type(data_type):
    f = Figlet(font='big')  # Using a different font for data type
    print('\n' + '*' * 80)
    print(colored(f.renderText(data_type.upper()), 'magenta'))
    print('*' * 80 + '\n')
    print('index: ', i)


def call_ollama(prompt: str) -> str:
    #print_step(1, "Calling Ollama API")
    #print(colored("Prompt:", 'green'))
    #print(prompt)
    completion = ollama.generate(model=MODEL_NAME,
                                 prompt=prompt,
                                 options={"temperature": TEMPERATURE})
    #print(colored("\nResponse:", 'green'))
    #print(completion['response'])
    return completion['response']


def split_into_sentences(text: str) -> List[str]:
    """Helper function to split text into sentences."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text)


def extract_statements(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Extract statements from the given text using Ollama, processing in chunks based on sentence count for better performance."""
    print_step(1, "Extracting Statements")

    # Split the text into sentences
    sentences = split_into_sentences(text)

    # Group the sentences into chunks of the specified size
    chunks = [sentences[i:i + sentences_per_chunk] for i in range(0, len(sentences), sentences_per_chunk)]

    all_statements = []

    for idx, chunk in enumerate(chunks):
        print(colored(f"Processing chunk {idx + 1}/{len(chunks)}", 'blue'))
        prompt = STATEMENT_EXTRACTION_PROMPT.format(text=chunk)
        print(colored("Prompt:", 'green'))
        print(prompt)

        # Call Ollama on each chunk
        response = call_ollama(prompt)

        # Extract and clean statements
        statements = [s.strip() for s in response.split('\n') if s.strip()]

        print(colored("\nExtracted Statements from chunk:", 'green'))
        for s in statements:
            print(f"- {s}")

        all_statements.extend(statements)

    print(colored("\nAll Extracted Statements:", 'green'))
    for s in all_statements:
        print(f"- {s}")

    return all_statements


def parse_constrained(response: str, metric: str) -> Dict[str, List[str]]:
    """Parse Ollama response using constrained generation."""
    # This would require implementing the Outlines library or a similar approach
    # For simplicity, we'll use a placeholder implementation
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"Failed to parse constrained generation output for {metric}.")
        return {label: [] for label in JSON_SCHEMAS[metric]["properties"].keys()}


def label_statements(summary_statements: List[str], original_statements: List[str], original_text: str, metric: str) -> \
        Dict[str, List[str]]:
    print_step(2, f"Labeling Statements for {metric.capitalize()}")
    if metric == "correctness":
        prompt = CORRECTNESS_LABELING_PROMPT.format(
            summary_statements="\n".join(summary_statements),
            original_statements="\n".join(original_statements)
        )
    else:  # faithfulness
        prompt = FAITHFULNESS_LABELING_PROMPT.format(
            summary_statements="\n".join(summary_statements),
            original_text=original_text
        )
    print(colored("Prompt:", 'green'))
    print(prompt)

    response = call_ollama(prompt)

    print(colored("\nLabels:", 'green'))
    print(response)

    if PARSING_STRATEGY == "regex":
        return parse_regex(response, metric)
    else:
        return parse_constrained(response, metric)


def parse_regex(response: str, metric: str) -> Dict[str, List[str]]:
    print_step(3, "Parsing with Regex")
    labels = {label: [] for label in REGEX_PATTERNS[metric].keys()}
    for line in response.split('\n'):
        for label, pattern in REGEX_PATTERNS[metric].items():
            if re.search(pattern, line):
                labels[label].append(line)

    print(colored("Parsed Results:", 'green'))
    for label, statements in labels.items():
        print(f"\n{label}:")
        for s in statements:
            print(f"- {s}")
    return labels


def calculate_metrics(labels: Dict[str, List[str]], metric: str) -> Dict[str, float]:
    print_step(4, f"Calculating {metric.capitalize()} Metrics")
    if metric == "correctness":
        tp = len(labels["TP"])
        fp = len(labels["FP"])
        fn = len(labels["FN"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        results = {"precision": precision, "recall": recall, "f1": f1}
    else:  # faithfulness
        passed = len(labels["PASSED"])
        failed = len(labels["FAILED"])
        precision = passed / (passed + failed) if (passed + failed) > 0 else 0
        results = {"precision": precision}

    print(colored("Calculated Metrics:", 'green'))
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    return results


def visualize_results_old(results: Dict[str, float], metric: str):
    print_step(5, "Visualizing Results")
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title(f"{metric.capitalize()} Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def evaluate_summary(summary: str, original_text: str) -> Dict[str, Dict[str, float]]:
    print_step(0, "Starting Evaluation")
    summary_statements = extract_statements(summary)
    original_statements = extract_statements(original_text)

    correctness_labels = label_statements(summary_statements=summary_statements,
                                          original_statements=original_statements, original_text=original_text,
                                          metric="correctness")

    correctness_metrics = calculate_metrics(correctness_labels, "correctness")

    #faithfulness_labels = label_statements(summary_statements, original_statements, original_text, "faithfulness")
    #faithfulness_metrics = calculate_metrics(faithfulness_labels, "faithfulness")

    results = {
        "original_statements": original_statements,
        "summary_statements": summary_statements,
        "correctness_labels": correctness_labels,
        "correctness": correctness_metrics,
        #"faithfulness": faithfulness_metrics
    }

    #for metric, metrics in results.items():
    visualize_results(correctness_metrics, "correctness")

    return results


def evaluate_all_types(original_text, summary, simplified_text, simplified_summary, relation_text, relation_summary,
                       relevance):
    print_data_type("Evaluating Original Text")
    original_results = evaluate_summary(summary, original_text)

    print_data_type("Evaluating Simplified Text")
    simplified_results = evaluate_summary(simplified_summary, simplified_text)
    if RELATION:
        print_data_type("Evaluating Relation Text")
        relation_results = evaluate_summary(relation_summary, relation_text)

        return {
            "original": original_results,
            "simplified": simplified_results,
            "relation": relation_results,
            "relevance": relevance
        }
    return {
        "original": original_results,
        "simplified": simplified_results,
        "relevance": relevance
    }


def compare_results(all_results):
    print_step(6, "Comparing Results")
    metrics = list(all_results[0]['original_labeling_statements'].keys())
    if RELATION:
        input_types = ['original', 'simplified', 'relation']
    else:
        input_types = ['original', 'simplified']
    aggregated_results = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        for input_type in input_types:
            for metric in metrics:
                value = result[f'{input_type}_labeling_statements'][metric]
                aggregated_results[input_type][metric].append(value)

    for input_type in input_types:
        print(colored(f"\n{input_type.capitalize()} Results:", 'green'))
        print(colored("Metrics:", 'green'))
        print(metrics)
        print(colored("aggregated_results:", 'green'))
        print(aggregated_results)
        for metric in metrics:
            values = aggregated_results[input_type][metric]
            avg_value = sum(values) / len(values)
            print(f"{metric.capitalize()}: {ascii_bar(avg_value)} (Avg: {avg_value:.4f})")


def save_to_csv(all_results, filename=CSV_FILE):
    if RELATION:
        input_types = ['original', 'simplified', 'relation']
    else:
        input_types = ['original', 'simplified']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = ["Index", "Sample", "Type", "Text", "Summary", "Extracted Statements Text",
                   "Extracted Statements Summary",
                   "Labeling Statements", "Label Counts", "Precision", "Recall", "F1", "Relevance"]
        writer.writerow(headers)

        for i, result in enumerate(all_results):
            #print(result)
            for input_type in input_types:
                metrics = result[f'{input_type}_labels']

                row = [
                    result['index'],
                    f"Sample {i + 1}",
                    input_type,
                    result[f'{input_type}_text'],
                    result[f'{input_type}_summary'],
                    "\n".join(result[f'{input_type}_extracted_statements']),
                    "\n".join(result[f'{input_type}_extracted_statements_summary']),
                    "\n ".join([f"{k}: {v}" for k, v in result[f'{input_type}_labeling_statements'].items()]),
                    ", ".join([f"{k}: {len(v)}" for k, v in result[f'{input_type}_labeling_statements'].items()]),
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    result['relevance']
                ]
                writer.writerow(row)

    print(f"Results saved to {filename}")


def save_to_pkl(all_results, filename='results.pkl'):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Load data
    with open("./save/qwen2.5:72b_dataset_modify.pkl", 'rb') as f:
        data = pickle.load(f)

    # Set random seed for reproducibility
    random.seed(42)
    num_samples = 2
    # Select a random index
    indexes = random.sample(range(len(data)), num_samples)
    all_results = []
    #indexes = [501, 1508]#, 1116, 1209]

    if RELATION:
        input_types = ['original', 'simplified', 'relation']
    else:
        input_types = ['original', 'simplified']
    i = 0
    for sample_data in data:
        print_step(i + 1, f"Processing Sample {i + 1} on {len(data)}")
        i += 1
        # Extract data for the selected index
        #sample_data = data[index]
        original_text = sample_data['text']
        original_summary = sample_data['summary']
        simplified_text = sample_data['simplified_text']
        simplified_summary = sample_data['simplified_summary']
        if RELATION:
            relation_text = sample_data['relation_text']
            relation_summary = sample_data['relation_summary']

        relevance = sample_data['relevance']

        print(colored("Index:", 'green'), i)
        print(colored("\nOriginal Text:", 'green'), original_text)
        print(colored("Summary:", 'green'), original_summary)
        print(colored("\nSimplified Text:", 'green'), simplified_text)
        print(colored("Simplified Summary:", 'green'), simplified_summary)
        if RELATION:
            print(colored("\nRelation Text:", 'green'), relation_text)
            print(colored("Relation Summary:", 'green'), relation_summary)

        print(colored("\nRelevance:", 'green'), relevance)

        # Evaluate all types
        if RELATION:
            results_all = evaluate_all_types(original_text, original_summary, simplified_text, simplified_summary,
                                             relation_text,
                                             relation_summary, relevance)
        else:
            results_all = evaluate_all_types(original_text, original_summary, simplified_text, simplified_summary,
                                             None,

                                            None, relevance)
        results = {'index': i}
        for input_type in input_types:
            results[input_type] = results_all[input_type]["correctness"]

        results['relevance'] = relevance
        # Add extracted statements and labels to results
        for input_type in input_types:
            text_var = f'{input_type}_text'
            summary_var = f'{input_type}_summary'

            results[text_var] = locals()[text_var]
            results[summary_var] = locals()[summary_var]
            results[f'{input_type}_extracted_statements'] = results_all[input_type][
                "original_statements"]  #"\n".join(extract_statements(results[summary_var]))
            results[f'{input_type}_extracted_statements_summary'] = results_all[input_type][
                "summary_statements"]  #"\n".join(extract_statements(results[summary_var]))
            # Capture the labeling statements
            results[f'{input_type}_labeling_statements'] = results_all[input_type]["correctness_labels"]
            results[f'{input_type}_labels'] = results_all[input_type]["correctness"]
        all_results.append(results)

        # Compare results across all samples
    #compare_results(all_results)
    # Save results to CSV
    #save_to_csv(all_results)
    #print(all_results)
    save_to_pkl(all_results, PKL_FILE_SAVE)
    print_step(num_samples + 2, "Final Results")
    #print(colored("All Evaluation Results:", 'green'))
    #print(json.dumps(all_results, indent=2))
