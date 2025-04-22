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

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Définir drapeau pour vérifier si OpenAI est disponible
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("Package 'openai' non disponible, les fonctionnalités OpenAI seront désactivées.")

# Définir drapeau pour vérifier si Ollama est disponible
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.warning("Package 'ollama' non disponible, les fonctionnalités Ollama seront désactivées.")


class GEvalMetric(TextMetric):
    """
    Implémentation de la métrique G-Eval pour l'évaluation de texte.
    
    G-Eval utilise des grands modèles de langage (LLMs) comme GPT-4 ou des modèles
    locaux via Ollama pour évaluer la qualité des textes générés selon différentes dimensions.
    """
    
    def __init__(self, 
                 dimension: str = "relevance", 
                 model_type: str = "openai",
                 model_name: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0001,
                 max_tokens: int = 10,
                 ollama_base_url: str = "http://localhost:11434",
                 prompt_template: Optional[str] = None,
                 n_responses: int = 3,
                 parse_numeric: bool = True,
                 verbose: bool = False,
                 debug_mode: bool = False,
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
            debug_mode: Mode de débogage détaillé pour diagnostiquer les problèmes
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
        
        # Configuration du niveau de logging en fonction du mode debug
        if debug_mode:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            logger.debug("Mode debug activé pour G-Eval")
        
        # Configuration des clients
        if model_type == "ollama":
            try:
                self.ollama_client = ollama.Client(host=ollama_base_url)
                logger.info(f"G-Eval initialisé avec Ollama et le modèle {model_name}")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client Ollama: {e}")
                raise

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
        self.debug_mode = debug_mode
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.ollama_base_url = ollama_base_url
        self.ollama_client = self.ollama_client if model_type == "ollama" else None

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
        if self.model_type == "openai":
            if api_key:
                openai.api_key = api_key
                logger.info(f"G-Eval initialisé avec OpenAI et le modèle {model_name}")
            elif not openai.api_key:
                logger.warning("Aucune clé API OpenAI fournie et aucune clé n'est définie dans l'environnement.")
                if debug_mode:
                    logger.debug("Vérifiez la configuration de la clé API OpenAI.")
        
        # Test des connexions en mode debug
        if debug_mode:
            self._test_connection()

    def _test_connection(self):
        """
        Teste la connexion selon le type de modèle choisi.
        
        En mode debug, cette fonction vérifie que la connexion à l'API
        (OpenAI ou Ollama) fonctionne correctement avec le modèle spécifié.
        """
        if self.model_type == "openai":
            self._test_openai_connection()
        elif self.model_type == "ollama":
            self._test_ollama_connection()
    
    def _test_openai_connection(self):
        """
        Teste la connexion à l'API OpenAI et vérifie si le modèle est disponible.
        """
        logger.debug("Test de connexion à l'API OpenAI...")
        
        try:
            # Vérifier si la clé API est configurée
            if not openai.api_key:
                logger.warning("Aucune clé API OpenAI définie.")
                return
            
            # Test simple d'appel API
            test_prompt = "Respond with 'OK' if you can read this."
            logger.debug(f"Test OpenAI avec le prompt: '{test_prompt}'")
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "system", "content": test_prompt}],
                max_tokens=5,
                temperature=0
            )
            
            logger.debug(f"Réponse du test OpenAI: {response.choices[0].message.content}")
            logger.debug(f"Test de connexion à OpenAI réussi")
            
        except Exception as e:
            logger.error(f"Erreur lors du test de connexion à OpenAI: {e}")
            if self.debug_mode:
                logger.debug(f"Type d'erreur: {type(e).__name__}")
                logger.debug(f"Détails: {str(e)}")
    
    def _test_ollama_connection(self):
        """
        Teste la connexion à Ollama et vérifie si le modèle est disponible.
        """
        logger.debug(f"Test de connexion à Ollama ({self.ollama_base_url})...")
        
        try:
            # Vérifier si le modèle est disponible
            models = ollama.list()
            logger.debug(f"Modèles disponibles sur Ollama: {models}")
            
            model_names = [model.get('name') for model in models.get('models', [])]
            if self.model_name not in model_names:
                logger.warning(f"Le modèle '{self.model_name}' n'est pas disponible dans Ollama. "
                             f"Modèles disponibles: {model_names}")
            else:
                logger.debug(f"Le modèle '{self.model_name}' est disponible dans Ollama.")
            
            # Test de génération simple
            test_prompt = "Respond with 'OK' if you can read this."
            logger.debug(f"Test Ollama avec le prompt: '{test_prompt}'")
            
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=test_prompt,
                options={"temperature": 0}
            )
            
            logger.debug(f"Réponse du test Ollama: {response.get('response', '')}")
            logger.debug(f"Test de connexion à Ollama réussi")
            
        except Exception as e:
            logger.error(f"Erreur lors du test de connexion à Ollama: {e}")
            if self.debug_mode:
                logger.debug(f"Type d'erreur: {type(e).__name__}")
                logger.debug(f"Détails: {str(e)}")
    
    def _load_default_prompt(self):
        """
        Charge le prompt par défaut pour la dimension spécifiée depuis le fichier approprié.
        En cas d'échec, utilise un prompt par défaut générique.
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
                if self.verbose or self.debug_mode:
                    logger.info(f"Prompt chargé depuis {prompt_path}")
            else:
                # Prompt par défaut si le fichier n'est pas trouvé
                self.prompt_template = (
                    f"Évaluez la qualité de ce texte sur une échelle de {self.min_score} à {self.max_score} "
                    f"pour la dimension '{self.dimension}' ({self.dimension_info.get(self.dimension, {}).get('description', '')}). "
                    f"\n\nTexte source:\n{{Document}}\n\nTexte à évaluer:\n{{Summary}}\n\n"
                    f"Indiquez simplement un score entre {self.min_score} et {self.max_score}."
                )
                logger.warning(f"Fichier de prompt non trouvé: {prompt_path}. Utilisation d'un prompt par défaut.")
                if self.debug_mode:
                    logger.debug(f"Prompt par défaut utilisé: {self.prompt_template}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du prompt: {e}")
            if self.debug_mode:
                logger.debug(f"Détails de l'erreur: {str(e)}")
                logger.debug(f"Type d'erreur: {type(e).__name__}")
            
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
        
        if self.debug_mode:
            logger.debug(f"Prompt préparé:")
            logger.debug(f"{prompt}")
            #logger.debug(f"Début: {prompt[:100]}...")
            #if len(prompt) > 200:
            #    logger.debug(f"Fin: ...{prompt[-100:]}")
            
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
            if self.debug_mode:
                logger.debug("Parsing numérique désactivé, utilisation de la valeur par défaut")
            return (self.min_score + self.max_score) / 2  # Valeur par défaut
        
        try:
            # Rechercher un nombre dans la réponse (entier ou décimal)
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                if self.debug_mode:
                    logger.debug(f"Score extrait: {score}")
                
                # Vérifier que le score est dans la plage attendue
                if self.min_score <= score <= self.max_score:
                    return score
                else:
                    # Normaliser le score à la plage attendue si hors limites
                    normalized_score = max(self.min_score, min(self.max_score, score))
                    if self.debug_mode:
                        logger.debug(f"Score hors limites normalisé: {normalized_score}")
                    return normalized_score
            else:
                if self.debug_mode:
                    logger.debug(f"Aucun score numérique trouvé dans la réponse: {response}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction du score: {e}")
            if self.debug_mode:
                logger.debug(f"Réponse problématique: {response}")
                logger.debug(f"Type d'erreur: {type(e).__name__}")
                logger.debug(f"Détails: {str(e)}")
        
        # Valeur par défaut en cas d'échec de l'extraction
        default_score = (self.min_score + self.max_score) / 2
        if self.debug_mode:
            logger.debug(f"Utilisation de la valeur par défaut: {default_score}")
        return default_score
    
    def _get_scores_from_openai(self, prompt: str) -> List[float]:
        """
        Obtenir des scores en utilisant le modèle OpenAI.
        
        Args:
            prompt: Prompt à envoyer au modèle
            
        Returns:
            Liste de scores
        """
        scores = []
        errors = []
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose or self.debug_mode:
                    logger.info(f"Envoi de la requête à OpenAI (tentative {attempt+1}/{self.max_retries})")
                
                start_time = time.time()
                
                # Utiliser l'API OpenAI pour générer des réponses
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n_responses
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if self.debug_mode:
                    logger.debug(f"Temps de réponse OpenAI: {elapsed_time:.2f} secondes")
                
                # Extraire les scores de chaque réponse
                for choice in response.choices:
                    content = choice.message.content
                    score = self._parse_score(content)
                    scores.append(score)
                    
                    if self.debug_mode:
                        logger.debug(f"Réponse brute: {content}")
                        logger.debug(f"Score extrait: {score}")
                
                # Limiter la fréquence des requêtes pour éviter les erreurs de rate limit
                if self.n_responses > 1 and attempt < self.max_retries - 1:
                    time.sleep(0.5)
                
                return scores
            
            except Exception as e:
                error_msg = f"Tentative {attempt+1}/{self.max_retries} échouée: {e}"
                logger.warning(error_msg)
                errors.append({"attempt": attempt+1, "error": str(e), "type": type(e).__name__})
                
                if "rate limit" in str(e).lower():
                    # Attendre plus longtemps en cas d'erreur de rate limit
                    wait_time = self.retry_delay * 5
                    logger.warning(f"Rate limit dépassé, pause de {wait_time} secondes")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Échec de la requête OpenAI après {self.max_retries} tentatives")
                    if self.debug_mode:
                        logger.debug(f"Erreurs: {json.dumps(errors, indent=2)}")
                    
                    # Renvoyer une valeur par défaut en cas d'échec
                    default_score = (self.min_score + self.max_score) / 2
                    return [default_score] * self.n_responses
    
    def _get_scores_from_ollama(self, prompt: str) -> List[float]:
        """
        Obtenir des scores en utilisant un modèle via Ollama.
        
        Args:
            prompt: Prompt à envoyer au modèle
            
        Returns:
            Liste de scores
        """
        scores = []
        errors = []
        
        for i in range(self.n_responses):
            if self.debug_mode:
                logger.debug(f"Génération de la réponse {i+1}/{self.n_responses}")
            
            for attempt in range(self.max_retries):
                try:
                    if self.verbose or self.debug_mode:
                        logger.info(f"Envoi de la requête à Ollama (réponse {i+1}, tentative {attempt+1}/{self.max_retries})")
                    
                    start_time = time.time()
                    
                    # Requête à l'API Ollama
                    response = self.ollama_client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": self.temperature}
                    )
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    if self.debug_mode:
                        logger.debug(f"Temps de réponse Ollama: {elapsed_time:.2f} secondes")
                        logger.debug(f"Réponse brute: {response.get('response', '')}")

                    # Extraire le score de la réponse
                    response_text = response.get('response', '')
                    score = self._parse_score(response_text)
                    scores.append(score)
                    
                    if self.debug_mode:
                        logger.debug(f"Score extrait: {score}")
                    
                    # Pause courte entre les requêtes pour éviter de surcharger Ollama
                    if i < self.n_responses - 1:
                        time.sleep(0.2)
                    
                    break  # Sortir de la boucle de tentatives si réussi
                    
                except Exception as e:
                    error_msg = f"Tentative {attempt+1}/{self.max_retries} échouée: {e}"
                    logger.warning(error_msg)
                    errors.append({"response": i+1, "attempt": attempt+1, "error": str(e), "type": type(e).__name__})
                    
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Nouvelle tentative dans {self.retry_delay} secondes...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Échec de la requête Ollama après {self.max_retries} tentatives")
                        # Ajouter une valeur par défaut en cas d'échec
                        default_score = (self.min_score + self.max_score) / 2
                        scores.append(default_score)
                        
                        if self.debug_mode:
                            logger.debug(f"Utilisation de la valeur par défaut: {default_score}")
        
        if self.debug_mode and errors:
            logger.debug(f"{len(errors)} erreurs rencontrées lors des requêtes Ollama:")
            logger.debug(f"Erreurs: {json.dumps(errors, indent=2)}")
        
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
                - debug_sample: Nombre d'exemples à traiter en mode debug (défaut: tous)
                - use_sources: Utiliser les sources au lieu des références si True
                
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Récupérer les paramètres optionnels
        debug_sample = kwargs.get('debug_sample', None)
        use_sources = kwargs.get('use_sources', False)
        
        # Pour les dimensions qui nécessitent un texte source
        if self.dimension in ["consistency", "coherence", "relevance"]:
            if sources is None:
                if self.verbose or self.debug_mode:
                    logger.info(f"La dimension '{self.dimension}' nécessite des textes source. Utilisation des références comme sources.")
                eval_sources = references
            else:
                if len(sources) != len(candidates):
                    raise ValueError(f"Le nombre de sources ({len(sources)}) ne correspond pas "
                                   f"au nombre de candidats ({len(candidates)})")
                eval_sources = sources
                if self.debug_mode:
                    logger.debug("Utilisation des sources explicites")
        else:
            # Pour les dimensions qui n'utilisent pas de source (comme fluency)
            eval_sources = [None] * len(candidates)
            if self.debug_mode:
                logger.debug(f"La dimension '{self.dimension}' n'utilise pas de texte source")
        
        # Limiter le nombre d'exemples en mode debug si spécifié
        if debug_sample is not None and 0 < debug_sample < len(candidates):
            if self.debug_mode:
                logger.debug(f"Mode debug: traitement de {debug_sample} exemples sur {len(candidates)}")
            candidates = candidates[:debug_sample]
            references = references[:debug_sample]
            eval_sources = eval_sources[:debug_sample]
        
        individual_scores = []
        all_responses = []
        errors = []
        processed_count = 0
        
        # Traiter chaque exemple
        for i, (candidate, reference, source) in enumerate(zip(candidates, references, eval_sources)):
            try:
                if self.verbose or self.debug_mode:
                    logger.info(f"Traitement de l'exemple {i+1}/{len(candidates)}")
                
                if self.debug_mode:
                    logger.debug(f"Candidat {i+1}: {candidate[:100]}...")
                    logger.debug(f"Référence {i+1}: {reference[:100]}...")
                    if source:
                        logger.debug(f"Source {i+1}: {source[:100]}...")
                
                # Utiliser la source appropriée pour l'évaluation
                eval_source = source if source is not None else reference
                
                # Préparer le prompt
                prompt = self._prepare_prompt(eval_source, candidate)
                
                # Obtenir les scores selon le type de modèle
                if self.model_type == "openai":
                    example_scores = self._get_scores_from_openai(prompt)
                else:  # ollama
                    example_scores = self._get_scores_from_ollama(prompt)
                
                # Transformer les scores de la plage [min_score, max_score] à [0, 1]
                normalized_scores = [(score - self.min_score) / (self.max_score - self.min_score) for score in example_scores]
                
                # Calculer le score moyen pour cet exemple
                avg_score = np.mean(normalized_scores) if normalized_scores else 0.5
                
                # Stocker les résultats
                individual_scores.append(avg_score)
                all_responses.append(example_scores)
                
                if self.debug_mode:
                    logger.debug(f"Scores bruts pour l'exemple {i+1}: {example_scores}")
                    logger.debug(f"Scores normalisés: {normalized_scores}")
                    logger.debug(f"Score moyen: {avg_score}")
                
                processed_count += 1
                
            except Exception as e:
                error_msg = f"Erreur lors du traitement de l'exemple {i+1}: {e}"
                logger.error(error_msg)
                errors.append({"index": i, "error": str(e), "error_type": type(e).__name__})
                
                # Ajouter un score par défaut pour éviter les incohérences
                individual_scores.append(0.5)  # Valeur médiane normalisée par défaut
                all_responses.append([])
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores) if individual_scores else 0.5
        
        # Construire le résultat final
        result = {
            'score': global_score,
            'individual_scores': individual_scores,
            'all_responses': all_responses,
            'dimension': self.dimension,
            'model': f"{self.model_type}:{self.model_name}",
            'score_range': {'min': self.min_score, 'max': self.max_score}
        }
        
        # Ajouter des métriques de diagnostic en mode debug
        if self.debug_mode:
            result['debug_info'] = {
                'processed_examples': processed_count,
                'total_examples': len(candidates),
                'success_rate': processed_count / len(candidates) if candidates else 0,
                'errors': errors if errors else []
            }
            logger.debug(f"Informations de débogage: {len(errors)} erreurs sur {len(candidates)} exemples")
            
        # Résumé d'exécution
        if self.verbose or self.debug_mode:
            success_rate = processed_count / len(candidates) if len(candidates) > 0 else 0
            logger.info(f"Évaluation G-Eval terminée: {processed_count}/{len(candidates)} exemples traités avec succès ({success_rate:.1%})")
            if errors:
                logger.warning(f"{len(errors)} erreurs rencontrées pendant l'évaluation")
            
        return result