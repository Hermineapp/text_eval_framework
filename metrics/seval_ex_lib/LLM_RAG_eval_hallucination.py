import json
import random
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from LLM_RAG_Eval_print_compare_StSummary_with_text_all_dataset import evaluate_summary, call_ollama
# Assuming we have the same data reading setup
import pickle
import tqdm
import os
import ast

# Loading the existing data from a file
with open("./save/qwen2.5:72b_dataset_modify.pkl", 'rb') as f:
    data = pickle.load(f)


def generate_hallucination(prompt, hallucination_type):
    """
    Function to generate hallucinations of specified type.
    """
    if hallucination_type == "entity_replacement":
        prompt = f"Replace named entities with incorrect ones in the following summary: {prompt}"
    elif hallucination_type == "incorrect_event":
        prompt = f"Modify the sequence of events in the following summary by adding incorrect details everywhere: {prompt}"
    elif hallucination_type == "fictitious_details":
        prompt = f"Add a fictitious but plausible detail everywhere to the following summary: {prompt}"
    else:
        raise ValueError("Unsupported hallucination type.")

    hallucinated_summary = call_ollama(prompt)

    return hallucinated_summary

# check if ./save/hallucinated_scores.csv exist
if not "hallucinated_scores.csv" in os.listdir("./save"):
    # Split the data into three equal parts
    num_samples = len(data)
    num_per_type = num_samples // 3

    hallucination_types = ["entity_replacement", "incorrect_event", "fictitious_details"]
    hallucinated_data = []

    # Create hallucinations for each type
    for i, hallucination_type in enumerate(hallucination_types):
        start_idx = i * num_per_type
        end_idx = (i + 1) * num_per_type if i != 2 else num_samples  # handle remainder for the last batch

        for idx in range(start_idx, end_idx):
            original_summary = data[idx]['summary']
            hallucinated_summary = generate_hallucination(original_summary, hallucination_type)

            # Run your existing evaluation function on the hallucinated summary
            hallucinated_score = evaluate_summary(hallucinated_summary, data[idx]['text'])

            # Store the results
            hallucinated_data.append({
                "index": idx,
                "hallucination_summary": hallucinated_summary,
                "hallucination_type": hallucination_type,
                "hallucinated_score": hallucinated_score
            })



    # Save the new hallucination scores
    hallucinated_scores_df = pd.DataFrame(hallucinated_data)
    hallucinated_scores_df.to_csv('./save/hallucinated_scores.csv', index=False)

else:
    hallucinated_scores_df = pd.read_csv('./save/hallucinated_scores.csv')

# Loading previous scores
FILE_PATH = "./save/St_compare_Sum_with_text_with_faith.pkl"
with open(FILE_PATH, 'rb') as f:
    data = pickle.load(f)

previous_scores_df = pd.DataFrame(data)
# Faithfullnes results : original_faithfulness-> faithfulness_f1
# correctness results : original_labels-> f1

# Comparison with previous scores
comparison_results = []

for _, row in hallucinated_scores_df.iterrows():
    row_index = row['index']
    #original_score = previous_scores_df.loc[previous_scores_df['index'] == row_index]
    original_score = previous_scores_df.iloc[row_index]
    #print(original_score['original_labels']['f1'])
    print('print row_index and content')
    print(row_index)

    # print the exact row_index
    print(type(row['hallucinated_score']))
    if isinstance(row['hallucinated_score'], str):
        # convert string to dictionary
        dic_row = ast.literal_eval(row['hallucinated_score'])


    else:
        dic_row = row['hallucinated_score']
    #print(previous_scores_df.loc[previous_scores_df['index'] == row_index])
    print("-" * 50)

    original_score_corr = original_score['original_labels']['f1']
    original_score_faith = original_score['original_faithfulness']['faithfulness_f1']

    add_comp = {
        "index": row['index'],
        "hallucination_type": row['hallucination_type'],
        "original_score_corr": original_score_corr,
        "original_score_faith": original_score_faith,
        "hallucinated_score_corr": dic_row['correctness']['f1'],
        "hallucinated_score_faith": dic_row['faithfulness']['faithfulness_f1'],
        "score_difference_corr": dic_row['correctness']['f1'] - original_score_corr,
        "score_difference_faith": dic_row['faithfulness']['faithfulness_f1'] - original_score_faith
    }
    print(add_comp)
    comparison_results.append(add_comp)
    exit()

# Save the comparison results
comparison_results_df = pd.DataFrame(comparison_results)
comparison_results_df.to_csv('./save_csv/comparison_scores.csv', index=False)
print("Hallucination generation and comparison completed.")
print(comparison_results_df[:10])
