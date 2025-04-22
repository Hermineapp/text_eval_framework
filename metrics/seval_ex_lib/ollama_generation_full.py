from datasets import load_dataset
from tqdm import tqdm
import ollama
import pickle

def get_preprompt_simplification():
    return """I need to simplify the sentences in the following text to enhance readability and comprehension. Please follow these detailed instructions to achieve the best possible results:

        Do not add other information and do not talk. Do not add "Here are the ..." sentence. Only simplify the sentences in the text.
        Simplify Vocabulary: Replace complex words and phrases with simpler, more commonly used synonyms without changing the original meaning.
        Shorten Sentences: Break down long, complex sentences into shorter, simpler ones while preserving the logical flow and connections between ideas.
        Clarify Structure: Simplify grammatical structures by using straightforward sentence constructions. Avoid using passive voice, subordinate clauses, and convoluted expressions.
        Preserve Meaning: Ensure that the essential meaning and information of the original text are maintained in the simplified version.
        Enhance Readability: Focus on making the text more accessible to a wider audience, including non-native speakers and individuals with varying reading skills.
        Avoid use pronoun
         like "he/she/it/they" and use the name of the entity instead.
        Examples and Clarifications:
        Example Text: 'The comprehensive analysis of the dataset, which encompasses a myriad of variables and intricate relationships, necessitates a profound understanding of statistical methodologies.'
        Example Simplification: 'The detailed study of the data, which includes many variables and complex relationships, requires a deep understanding of statistics.'
        Here is the text:
        """


def get_preprompt_relation():
    return """I need to transform the following text into a structured format of '{entity 1} {relation} {entity 2}'. Please follow these detailed instructions to ensure the best possible results:
        Do not add other information and do not talk. Do not add "Here are the structured entity-relation pairs:" sentence. Only follow instruction.
        Identify Entities: Detect all significant entities in the text. Entities can be people, organizations, locations, dates, specific terms, or concepts.
        Identify Relationships: Determine the relationships between these entities. Relationships describe how one entity is connected to another, such as actions, properties, or associations.
        Format Output: For each pair of entities and their relationship, format them in the structure '{entity 1} {relation} {entity 2}'. Ensure each entity-relation pair is on a new line.
        Context Preservation: Ensure that the meaning and context of the original text are preserved in the extracted relationships.
        Examples and Clarifications:
        Example Text: "Alice, a biologist at BioCorp, discovered a new species of plant called FloraX in the Amazon rainforest in 2021. FloraX can absorb pollutants from the soil."
        Example Output:
        Alice works at BioCorp.
        Alice discovered FloraX.
        FloraX is a species of plant.
        FloraX located in Amazon rainforest.
        FloraX discovered in 2021.
        FloraX can_absorb pollutants.
        Here is the text:"""


dataset = load_dataset('mteb/summeval')

# Préparer la nouvelle liste de données
new_dataset = []

# Associer chaque 'text' avec chaque 'machine_summary'
"""for entry in dataset['test']:
    text = entry['text']
    for i in range(len(entry['machine_summaries'])):
        new_dataset.append({
            'text': text,
            'summary': entry['machine_summaries'][i],
            'relevance': entry['relevance'][i],
            'coherence': entry['coherence'][i],
            'fluency': entry['fluency'][i],
            'consistency': entry['consistency'][i]
        })"""

# Si vous voulez convertir la liste en un objet Dataset de Hugging Face

dataset_modify = []
models = ['llama3.1:70b']#['llama3.1:8b', 'llama3.1:70b']

for model in models:
    print(f"Model: {model}")
    for entry in tqdm(dataset['test']):
        print(f"len(dataset_modify): {len(dataset_modify)}")

        text = entry['text']

        # Generate a completion
        completion = ollama.generate(model=model,
                                     prompt=get_preprompt_simplification() + text + "\nSimplified text:\n")
        # print(get_preprompt_simplification() + data['text'] + "\nSimplified text:\n")
        # print("Response:", completion['response'])
        simplified_text = completion['response']

        completion = ollama.generate(model=model, prompt=get_preprompt_relation() + text +
                                                         "\nStructured entity-relation pairs:\n")

        # print(get_preprompt_relation() + data['text'] + "\nStructured entity-relation pairs:\n")
        # print("Response:", completion['response'])
        relation_text = completion['response']


        for i in range(len(entry['machine_summaries'])):
            data = {
                'text': text,
                'summary': entry['machine_summaries'][i],
                'relevance': entry['relevance'][i],
                'coherence': entry['coherence'][i],
                'fluency': entry['fluency'][i],
                'consistency': entry['consistency'][i]
            }

            # Generate a completion
            completion = ollama.generate(model=model, prompt=get_preprompt_simplification() + data['summary'] + "\nSimplified text:\n")
            #print(get_preprompt_simplification() + data['text'] + "\nSimplified text:\n")
            #print("Response:", completion['response'])
            simplified_summary = completion['response']

            completion = ollama.generate(model=model, prompt=get_preprompt_relation() + data['summary'] + "\nStructured entity-relation pairs:\n")
            #print(get_preprompt_relation() + data['text'] + "\nStructured entity-relation pairs:\n")
            #print("Response:", completion['response'])
            relation_summary = completion['response']

            dataset_modify.append({
                'text': data['text'],
                'summary': data['summary'],
                'relevance': data['relevance'],
                'coherence': data['coherence'],
                'fluency': data['fluency'],
                'consistency': data['consistency'],
                'simplified_text': simplified_text,
                'relation_text': relation_text,
                'simplified_summary': simplified_summary,
                'relation_summary': relation_summary
            })

    # save the little_dataset_modify in pickle file
    with open(f'{model}_dataset_modify.pkl', 'wb') as f:
        pickle.dump(dataset_modify, f)
    print(f"Saved {model}_dataset_modify.pkl")