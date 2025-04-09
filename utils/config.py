"""
Module pour gérer la configuration du framework d'évaluation.
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional
import jsonschema


# Schéma JSON pour valider la configuration
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object"}
                }
            }
        },
        "correlation": {
            "type": "object",
            "properties": {
                "methods": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "human_scores_path": {"type": "string"},
                        "references_path": {"type": "string"},
                        "candidates_path": {"type": "string"}
                    },
                    "required": ["human_scores_path"]
                },
                "visualization": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "output_dir": {"type": "string"},
                        "plot_individual_metrics": {"type": "boolean"}
                    }
                }
            }
        },
        "output": {
            "type": "object",
            "properties": {
                "report_dir": {"type": "string"},
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["json", "yaml", "csv"]}
                }
            }
        }
    },
    "required": ["metrics"]
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge une configuration à partir d'un fichier JSON ou YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dict: Configuration chargée
        
    Raises:
        ValueError: Si le format du fichier n'est pas supporté
    """
    ext = os.path.splitext(config_path)[1].lower()
    
    with open(config_path, 'r') as f:
        if ext in ('.yaml', '.yml'):
            config = yaml.safe_load(f)
        elif ext == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Format de configuration non supporté: {ext}")
    
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Valide une configuration selon le schéma défini.
    
    Args:
        config: Configuration à valider
        
    Returns:
        List: Liste des erreurs (vide si la configuration est valide)
    """
    try:
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        return []
    except jsonschema.exceptions.ValidationError as e:
        return [str(e)]


def create_default_config() -> Dict[str, Any]:
    """
    Crée une configuration par défaut.
    
    Returns:
        Dict: Configuration par défaut
    """
    return {
        "metrics": [
            {
                "name": "text_eval_framework.metrics.bleu.BLEUMetric",
                "params": {
                    "weights": [0.25, 0.25, 0.25, 0.25],
                    "smoothing": "method1"
                }
            },
            {
                "name": "text_eval_framework.metrics.rouge.ROUGEMetric",
                "params": {
                    "rouge_types": ["rouge1", "rouge2", "rougeL"],
                    "use_stemmer": True
                }
            }
        ],
        "correlation": {
            "methods": ["pearson", "spearman", "kendall"],
            "data": {
                "human_scores_path": "data/human_scores.csv"
            },
            "visualization": {
                "enabled": True,
                "output_dir": "results/plots",
                "plot_individual_metrics": True
            }
        },
        "output": {
            "report_dir": "results/reports",
            "formats": ["json", "csv"]
        }
    }


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Sauvegarde une configuration dans un fichier.
    
    Args:
        config: Configuration à sauvegarder
        output_path: Chemin de sortie
        
    Raises:
        ValueError: Si le format du fichier n'est pas supporté
    """
    ext = os.path.splitext(output_path)[1].lower()
    
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        if ext in ('.yaml', '.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif ext == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Format de configuration non supporté: {ext}")
