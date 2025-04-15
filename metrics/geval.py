"""
Module implémentant la métrique G-Eval pour l'évaluation de texte avec des LLMs.

G-Eval (Liu et al., 2023) est une approche d'évaluation de texte généré qui utilise des grands 
modèles de langage (LLMs) comme évaluateurs. Cette implémentation supporte à la fois les modèles
OpenAI via API et les modèles locaux via Ollama.

Référence:
    Liu, Yang, et al. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
    https://arxiv.org/abs/2303.16634
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
from core.metric_interface import TextMetric

# Définir drapeau pour vérifier si OpenAI est disponible
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# Définir drapeau pour vérifier si Ollama est disponible
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


class GEvalMetric(TextMetric):
    """
    Implémentation de la métrique G-Eval pour l'évaluation de texte.
    
    G-Eval utilise des grands modèles de langage (LLMs) comme GPT-4 ou des modèles
    locaux via Ollama pour évaluer la qualité des textes générés selon différentes dimensions.
    """
    
    def __init__(self, 
                 dimension: str = "relevance", 
                 model_type: str = "openai",
                 model_name: str = "gpt-4o",#"gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 10,
                 ollama_base_url: str = "http://localhost:11434",
                 prompt_template: Optional[str] = None,
                 n_responses: int = 3,
                 parse_numeric: bool = True,
                 verbose: bool = False,
                 retry_delay: int = 1,
                 max_retries: int = 3,
                 score_range: Optional[Dict] = None):
        """
        Initialise la métrique G-Eval.
        
        Args:
            dimension: Dimension d'évaluation ('coherence', 'consistency', 'fluency', 'relevance')
            model_type: Type de modèle à utiliser ('openai' ou 'ollama')
            model_name: Nom du modèle à utiliser
            api_key: Clé API pour OpenAI
            temperature: Température d'échantillonnage pour le modèle
            max_tokens: Nombre maximum de tokens pour la réponse
            ollama_base_url: URL de base pour l'API Ollama 
            prompt_template: Modèle de prompt personnalisé
            n_responses: Nombre de réponses à générer par requête
            parse_numeric: Extraire automatiquement les scores numériques des réponses
            verbose: Afficher les détails pendant l'évaluation
            retry_delay: Délai entre les tentatives en cas d'erreur (en secondes)
            max_retries: Nombre maximum de tentatives en cas d'erreur
            score_range: Plage de scores pour la dimension (ex: {'min': 1, 'max': 5})
            
        Raises:
            ImportError: Si les dépendances requises ne sont pas installées
        """
        # Valider la disponibilité des dépendances
        if model_type == "openai" and not _OPENAI_AVAILABLE:
            raise ImportError(
                "Le package 'openai' est requis pour utiliser GEvalMetric avec OpenAI. "
                "Installez-le avec 'pip install openai'."
            )
        elif model_type == "ollama" and not _OLLAMA_AVAILABLE:
            raise ImportError(
                "Le package 'ollama' est requis pour utiliser GEvalMetric avec Ollama. "
                "Installez-le avec 'pip install ollama'."
            )
        
        if model_type == "ollama":
            ollama_client = ollama.Client(host=ollama_base_url)

        # Informations sur la dimension (par défaut)
        self.dimension_info = {
            "coherence": {
                "description": "structure et cohérence du texte",
                "min_score": 1,
                "max_score": 5,
                "file": "coh_detailed.txt"
            },
            "consistency": {
                "description": "cohérence factuelle avec le document source",
                "min_score": 1,
                "max_score": 5,
                "file": "con_detailed.txt"
            },
            "fluency": {
                "description": "qualité linguistique et grammaticale",
                "min_score": 1,
                "max_score": 3,
                "file": "flu_detailed.txt"
            },
            "relevance": {
                "description": "pertinence du contenu par rapport à la source",
                "min_score": 1,
                "max_score": 5,
                "file": "rel_detailed.txt"
            }
        }
        
        # Paramètres de base
        self.dimension = dimension
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_responses = n_responses
        self.parse_numeric = parse_numeric
        self.verbose = verbose
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.ollama_base_url = ollama_base_url
        self.ollama_client = ollama_client if model_type == "ollama" else None

        # Définir la plage de scores
        if score_range:
            self.min_score = score_range.get("min", 1)
            self.max_score = score_range.get("max", 5)
        else:
            self.min_score = self.dimension_info.get(dimension, {}).get("min_score", 1)
            self.max_score = self.dimension_info.get(dimension, {}).get("max_score", 5)
        
        # Charger le prompt par défaut
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self._load_default_prompt()
            
        # Configuration de l'API OpenAI si nécessaire
        if self.model_type == "openai" and api_key:
            openai.api_key = api_key
    
    def _load_default_prompt(self):
        """
        Charge le prompt par défaut pour la dimension spécifiée.
        """
        try:
            # Construire le chemin vers le fichier de prompt
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file = self.dimension_info.get(self.dimension, {}).get("file", "coh_detailed.txt")
            prompt_path = os.path.join(current_dir, "geval", "prompts", "summeval", prompt_file)
            
            # Lire le prompt depuis le fichier
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self.prompt_template = f.read()
                if self.verbose:
                    logging.info(f"Prompt chargé depuis {prompt_path}")
            else:
                # Prompt par défaut si le fichier n'est pas trouvé
                self.prompt_template = (
                    f"Évaluez la qualité de ce texte sur une échelle de {self.min_score} à {self.max_score} "
                    f"pour la dimension '{self.dimension}' ({self.dimension_info.get(self.dimension, {}).get('description', '')}). "
                    f"\n\nTexte source:\n{{Document}}\n\nTexte à évaluer:\n{{Summary}}\n\n"
                    f"Indiquez simplement un score entre {self.min_score} et {self.max_score}."
                )
                logging.warning(f"Fichier de prompt non trouvé: {prompt_path}. Utilisation d'un prompt par défaut.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du prompt: {e}")
            # Prompt minimal de secours
            self.prompt_template = (
                f"Évaluez ce texte de {self.min_score} à {self.max_score} pour la {self.dimension}.\n"
                f"Source: {{Document}}\nTexte: {{Summary}}\nScore:"
            )
    
    def _prepare_prompt(self, source: str, candidate: str) -> str:
        """
        Prépare le prompt pour le modèle en remplaçant les variables.
        
        Args:
            source: Texte source
            candidate: Texte candidat à évaluer
            
        Returns:
            Prompt formaté
        """
        prompt = self.prompt_template
        prompt = prompt.replace("{{Document}}", source)
        prompt = prompt.replace("{{Summary}}", candidate)
        return prompt
    
    def _parse_score(self, response: str) -> float:
        """
        Extraire le score numérique de la réponse du modèle.
        
        Args:
            response: Réponse du modèle
            
        Returns:
            Score numérique
        """
        if not self.parse_numeric:
            return 0.0  # Valeur par défaut si on ne peut pas parser
        
        try:
            # Rechercher un nombre dans la réponse (entier ou décimal)
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                # Vérifier que le score est dans la plage attendue
                if self.min_score <= score <= self.max_score:
                    return score
                else:
                    # Normaliser le score à la plage attendue si hors limites
                    return max(self.min_score, min(self.max_score, score))
        except Exception as e:
            if self.verbose:
                logging.warning(f"Erreur lors de l'extraction du score: {e}")
        
        # Valeur par défaut en cas d'échec de l'extraction
        return (self.min_score + self.max_score) / 2
    
    def _get_scores_from_openai(self, prompt: str) -> List[float]:
        """
        Obtenir des scores en utilisant le modèle OpenAI.
        
        Args:
            prompt: Prompt à envoyer au modèle
            
        Returns:
            Liste de scores
        """
        for attempt in range(self.max_retries):
            try:
                # Utiliser l'API OpenAI pour générer des réponses
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n_responses
                )
                
                scores = []
                for choice in response.choices:
                    content = choice.message.content
                    score = self._parse_score(content)
                    scores.append(score)
                
                # Limiter la fréquence des requêtes pour éviter les erreurs de rate limit
                if self.n_responses > 1:
                    time.sleep(0.5)
                
                return scores
            
            except Exception as e:
                if self.verbose:
                    logging.warning(f"Tentative {attempt+1}/{self.max_retries} échouée: {e}")
                if "rate limit" in str(e).lower():
                    # Attendre plus longtemps en cas d'erreur de rate limit
                    time.sleep(self.retry_delay * 5)
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"Erreur lors de la requête OpenAI: {e}")
                    # Renvoyer une valeur par défaut en cas d'échec
                    return [(self.min_score + self.max_score) / 2] * self.n_responses
    
    def _get_scores_from_ollama(self, prompt: str) -> List[float]:
        """
        Obtenir des scores en utilisant un modèle via Ollama.
        
        Args:
            prompt: Prompt à envoyer au modèle
            
        Returns:
            Liste de scores
        """
        scores = []
        for _ in range(self.n_responses):
            for attempt in range(self.max_retries):
                try:
                    # Requête à l'API Ollama
                    print(f"Envoi de la requête à Ollama: {prompt}")
                    response = self.ollama_client.generate(
                        model=self.model_name,
                        prompt=prompt,
                    )
                    print(f"Réponse du modèle: {response}")

                    # Extraire le score de la réponse
                    response_text = response.get('response', '')
                    score = self._parse_score(response_text)
                    scores.append(score)
                    break
                except Exception as e:
                    if self.verbose:
                        logging.warning(f"Tentative {attempt+1}/{self.max_retries} échouée: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logging.error(f"Erreur lors de la requête Ollama: {e}")
                        # Ajouter une valeur par défaut en cas d'échec
                        scores.append((self.min_score + self.max_score) / 2)
        
        return scores
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        model_suffix = self.model_name.replace("-", "_").replace(".", "_")
        return f"geval_{self.dimension}_{model_suffix}"
    
    def compute(self, references: List[str], candidates: List[str], sources: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores G-Eval entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            sources: Liste de textes source (obligatoire pour certaines dimensions)
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Pour les dimensions qui nécessitent un texte source
        if self.dimension in ["consistency", "coherence", "relevance"] and sources is None:
            if self.verbose:
                logging.warning(f"La dimension '{self.dimension}' nécessite des textes source. Utilisation des références comme sources.")
            sources = references
        
        individual_scores = []
        all_responses = []
        
        for i, (candidate, reference) in enumerate(zip(candidates, references)):
            source = sources[i] if sources else reference
            
            # Préparer le prompt
            prompt = self._prepare_prompt(source, candidate)
            
            # Obtenir les scores selon le type de modèle
            if self.model_type == "openai":
                scores = self._get_scores_from_openai(prompt)
            else:  # ollama
                scores = self._get_scores_from_ollama(prompt)
            
            # Transformer les scores de la plage [min_score, max_score] à [0, 1]
            scores = [(score - self.min_score) / (self.max_score - self.min_score) for score in scores]
            # Calculer le score moyen pour cet exemple
            avg_score = np.mean(scores) if scores else (self.min_score + self.max_score) / 2
            individual_scores.append(avg_score)
            all_responses.append(scores)
            
            if self.verbose:
                logging.info(f"Exemple {i+1}/{len(candidates)}: Score = {avg_score} (moyenne de {scores})")
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        # Normalisation optionnelle des scores entre 0 et 1
        normalized_scores = []
        score_range = self.max_score - self.min_score
        if score_range > 0:
            normalized_scores = [(score - self.min_score) / score_range for score in individual_scores]
            normalized_global = (global_score - self.min_score) / score_range
        else:
            normalized_scores = [score / self.max_score for score in individual_scores]
            normalized_global = global_score / self.max_score
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'normalized_score': normalized_global,
            'normalized_individual_scores': normalized_scores,
            'all_responses': all_responses,
            'dimension': self.dimension,
            'model': f"{self.model_type}:{self.model_name}",
            'score_range': {'min': self.min_score, 'max': self.max_score}
        }