"""
Module pour charger et prétraiter les données d'évaluation.
"""

import csv
import json
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
import os


def load_text_pairs(filepath: str, 
                   reference_col: Optional[str] = None, 
                   candidate_col: Optional[str] = None,
                   delimiter: str = ',',
                   encoding: str = 'utf-8') -> Tuple[List[str], List[str]]:
    """
    Charge les paires de textes (référence, candidat) à partir d'un fichier.
    
    Args:
        filepath: Chemin vers le fichier
        reference_col: Nom de la colonne contenant les références
        candidate_col: Nom de la colonne contenant les candidats
        delimiter: Délimiteur pour les fichiers CSV
        encoding: Encodage du fichier
        
    Returns:
        Tuple: (références, candidats)
        
    Raises:
        ValueError: Si le format du fichier n'est pas supporté
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)
        
        # Déterminer automatiquement les colonnes si non spécifiées
        if reference_col is None:
            potential_ref_cols = [col for col in df.columns 
                                 if any(term in col.lower() 
                                      for term in ['reference', 'ref', 'source', 'original'])]
            reference_col = potential_ref_cols[0] if potential_ref_cols else df.columns[0]
            
        if candidate_col is None:
            potential_cand_cols = [col for col in df.columns 
                                  if any(term in col.lower() 
                                       for term in ['candidate', 'cand', 'target', 'hyp', 'hypothesis', 'generated'])]
            candidate_col = potential_cand_cols[0] if potential_cand_cols else df.columns[1]
        
        return df[reference_col].tolist(), df[candidate_col].tolist()
        
    elif ext == '.json':
        with open(filepath, 'r', encoding=encoding) as f:
            data = json.load(f)
            
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Déterminer automatiquement les clés
            if reference_col is None or candidate_col is None:
                keys = data[0].keys()
                ref_keys = [k for k in keys if any(term in k.lower() 
                                               for term in ['reference', 'ref', 'source', 'original'])]
                cand_keys = [k for k in keys if any(term in k.lower() 
                                                for term in ['candidate', 'cand', 'target', 'hyp', 'hypothesis', 'generated'])]
                
                reference_col = reference_col or (ref_keys[0] if ref_keys else list(keys)[0])
                candidate_col = candidate_col or (cand_keys[0] if cand_keys else list(keys)[1])
            
            return [item[reference_col] for item in data], [item[candidate_col] for item in data]
            
        elif isinstance(data, dict) and all(isinstance(data.get('references'), list) 
                                        and isinstance(data.get('candidates'), list)):
            return data['references'], data['candidates']
            
    elif ext in ('.txt', '.tsv'):
        references, candidates = [], []
        with open(filepath, 'r', encoding=encoding) as f:
            delim = '\t' if ext == '.tsv' else delimiter
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if len(row) >= 2:
                    references.append(row[0])
                    candidates.append(row[1])
        
        return references, candidates
    
    raise ValueError(f"Format de fichier non supporté: {ext}")


def load_eval_data(text_path: str, 
                  human_scores_path: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Charge les données complètes pour l'évaluation.
    
    Args:
        text_path: Chemin vers le fichier contenant les paires de textes
        human_scores_path: Chemin vers le fichier contenant les scores humains
        **kwargs: Paramètres supplémentaires pour load_text_pairs
        
    Returns:
        Dict: Dictionnaire contenant les références, candidats et scores humains
    """
    references, candidates = load_text_pairs(text_path, **kwargs)
    
    result = {
        'references': references,
        'candidates': candidates
    }
    
    if human_scores_path:
        # Détecter le format du fichier de scores
        ext = os.path.splitext(human_scores_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(human_scores_path)
            # Trouver la colonne de scores
            score_cols = [col for col in df.columns 
                         if any(term in col.lower() 
                              for term in ['score', 'rating', 'human', 'eval', 'judgment'])]
            score_col = score_cols[0] if score_cols else df.columns[0]
            result['human_scores'] = df[score_col].tolist()
            
        elif ext == '.json':
            with open(human_scores_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
                result['human_scores'] = data
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # Trouver la clé de score
                keys = data[0].keys()
                score_keys = [k for k in keys if any(term in k.lower() 
                                                for term in ['score', 'rating', 'human', 'eval', 'judgment'])]
                score_key = score_keys[0] if score_keys else list(keys)[0]
                result['human_scores'] = [item[score_key] for item in data]
            elif isinstance(data, dict) and 'scores' in data:
                result['human_scores'] = data['scores']
        
        elif ext == '.txt':
            with open(human_scores_path, 'r') as f:
                result['human_scores'] = [float(line.strip()) for line in f]
    
    return result
