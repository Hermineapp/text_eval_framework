"""
Module implémentant la métrique SEval-Ex pour l'évaluation de résumés de texte.

SEval-Ex est un framework d'évaluation qui décompose les résumés en déclarations atomiques
et mesure leur alignement factuel avec le texte source, offrant à la fois une bonne
corrélation avec le jugement humain et une évaluation explicable de la cohérence factuelle.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from core.metric_interface import TextMetric

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vérifier si Ollama est disponible
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

class SEvalExMetric(TextMetric):
    """
    Implémentation de la métrique SEval-Ex pour l'évaluation de résumés.
    
    SEval-Ex emploie un pipeline en deux étapes:
    1. Extraction de déclarations: Décompose le texte source et le résumé en déclarations atomiques
    2. Raisonnement de verdict: Catégorise les déclarations comme TP (True Positives), 
       FP (False Positives), ou FN (False Negatives)
    """
    
    def __init__(self, 
                 model_name: str = "qwen2.5:72b", 
                 method: str = "StSum_Text", 
                 chunk_size: int = 3,
                 max_text_length: int = 500,
                 temperature: float = 0.0001,
                 ollama_base_url: str = "http://localhost:11434",
                 verbose: bool = False,
                 debug_mode: bool = False,
                 max_retries: int = 3,
                 retry_delay: int = 2):
        """
        Initialise la métrique SEval-Ex.
        
        Args:
            model_name: Nom du modèle LLM à utiliser
            method: Méthode d'évaluation ('StSum_Text', 'Base', ou 'Chunked')
            chunk_size: Nombre de phrases par chunk pour préserver le contexte local
            max_text_length: Longueur maximale du texte avant application du chunking
            temperature: Température pour le modèle LLM
            ollama_base_url: URL de base pour l'API Ollama
            verbose: Afficher les détails pendant l'évaluation
            debug_mode: Mode de débogage détaillé pour diagnostiquer les problèmes
            max_retries: Nombre maximum de tentatives en cas d'erreur de connexion
            retry_delay: Délai entre les tentatives en secondes
            
        Raises:
            ImportError: Si le package ollama n'est pas installé
        """
        if not _OLLAMA_AVAILABLE:
            raise ImportError(
                "Le package 'ollama' est requis pour utiliser SEvalExMetric. "
                "Installez-le avec 'pip install ollama'."
            )
        
        self.model_name = model_name
        self.method = method
        self.chunk_size = chunk_size
        self.max_text_length = max_text_length
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configuration du niveau de logging en fonction du mode debug
        if self.debug_mode:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            logging.getLogger(__name__).info("Mode debug activé pour SEval-Ex")
        
        # Initialiser le client Ollama
        try:
            self.ollama_client = ollama.Client(host=ollama_base_url)
            logger.info(f"SEval-Ex initialisé avec le modèle {model_name} et la méthode {method}")
            logger.debug(f"Configuration complète: chunk_size={chunk_size}, " 
                       f"max_text_length={max_text_length}, temperature={temperature}, "
                       f"ollama_base_url={ollama_base_url}")
            
            # Vérifier la connexion Ollama
            if self.debug_mode:
                self._test_ollama_connection()
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation d'Ollama: {e}")
            raise
        
        # Définir les prompts d'évaluation
        self._define_prompts()
    
    def _define_prompts(self):
        """Définit les prompts utilisés pour l'extraction et l'évaluation des déclarations."""
        self.statement_extraction_prompt = """
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
        
        self.correctness_labeling_prompt = """
        Compare the following statements from the summary with the statements from the original text. Do not add any ponctuation or words. Do not justify your answer.
        Label each summary statement as:
        - TP (True Positive): If the statement appears in the summary and is directly supported by a statement from the original text.
        - FP (False Positive): If the statement appears in the summary but is not directly supported by a statement from the original text.
        - FN (False Negative): If it appears in the original text but does not support any statement from the summary.
        
        As you can see in the example bellow, first you have to concatenate the summary statements and the original text statements. 
        Then you have to label each statement as TP, FP or FN. Format as follow: VERDICT: TP, VERDICT: FP, VERDICT: FN
        
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
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "seval_ex"
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Divise le texte en chunks sémantiquement cohérents pour préserver le contexte local.
        
        Args:
            text: Le texte source à diviser en chunks
            
        Returns:
            Liste de chunks de texte
        """
        import re
        # Diviser le texte en phrases avec une expression régulière plus robuste
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
        
        # S'assurer que les phrases se terminent par une ponctuation appropriée
        sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]
        
        # Regrouper les phrases en chunks
        chunks = []
        for i in range(0, len(sentences), self.chunk_size):
            chunk = ' '.join(sentences[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _test_ollama_connection(self):
        """
        Teste la connexion à Ollama et vérifie si le modèle est disponible.
        Renvoie des informations détaillées en mode debug.
        """
        import time
        
        logger.debug("Test de connexion à Ollama...")
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
            logger.debug(f"Test de génération avec le prompt: '{test_prompt}'")
            
            response = ollama.generate(
                model=self.model_name,
                prompt=test_prompt,
                options={"temperature": 0}
            )
            
            logger.debug(f"Réponse du test: {response.get('response', '')}")
            
        except Exception as e:
            logger.error(f"Erreur lors du test de connexion à Ollama: {e}")
            raise

    def _call_ollama(self, prompt: str) -> str:
        """
        Appelle l'API Ollama avec le prompt spécifié.
        
        Args:
            prompt: Le prompt à envoyer à Ollama
            
        Returns:
            La réponse générée par Ollama
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                if self.verbose or self.debug_mode:
                    logger.info(f"Sending prompt to Ollama (attempt {attempt+1}/{self.max_retries})")
                    
                if self.debug_mode:
                    logger.debug(f"Prompt complet: {prompt}")
                elif self.verbose:
                    logger.info(f"Début du prompt: {prompt[:100]}...")
                
                start_time = time.time()
                completion = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={"temperature": self.temperature}
                )
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                if self.debug_mode:
                    logger.debug(f"Temps de génération: {elapsed_time:.2f} secondes")
                    logger.debug(f"Réponse complète: {completion['response']}")
                elif self.verbose:
                    logger.info(f"Received response from Ollama in {elapsed_time:.2f}s")
                    logger.info(f"Début de la réponse: {completion['response'][:100]}...")
                
                return completion['response']
                
            except Exception as e:
                if self.debug_mode:
                    logger.error(f"Error calling Ollama API (attempt {attempt+1}/{self.max_retries}): {e}")
                    
                if attempt < self.max_retries - 1:
                    logger.warning(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to call Ollama API after {self.max_retries} attempts: {e}")
                    if self.debug_mode:
                        logger.error(f"Dernière erreur: {str(e)}")
                        logger.error(f"Type d'erreur: {type(e).__name__}")
                    
                    # Renvoyer une réponse vide en cas d'échec final
                    return ""
    
    def _extract_statements(self, text: str) -> List[str]:
        """
        Extrait les déclarations atomiques du texte donné en utilisant le LLM.
        
        Args:
            text: Le texte source duquel extraire les déclarations
            
        Returns:
            Liste de déclarations atomiques extraites
        """
        # Si la méthode est "Chunked" et que le texte est long, le traiter par chunks
        if self.method == "Chunked" and len(text.split()) > self.max_text_length:
            chunks = self._split_into_chunks(text)
            all_statements = []
            
            for chunk in chunks:
                chunk_prompt = self.statement_extraction_prompt.format(text=chunk)
                response = self._call_ollama(chunk_prompt)
                statements = [s.strip() for s in response.split('\n') if s.strip()]
                all_statements.extend(statements)
            
            return all_statements
        else:
            # Traitement direct du texte entier
            prompt = self.statement_extraction_prompt.format(text=text)
            response = self._call_ollama(prompt)
            return [s.strip() for s in response.split('\n') if s.strip()]
    
    def _verify_statements(self, summary_statements: List[str], original_statements: List[str]) -> Dict[str, List[str]]:
        """
        Vérifie les déclarations du résumé par rapport aux déclarations du texte source.
        
        Args:
            summary_statements: Liste de déclarations extraites du résumé
            original_statements: Liste de déclarations extraites du texte source
            
        Returns:
            Dictionnaire de classification des déclarations en TP, FP, et FN
        """
        prompt = self.correctness_labeling_prompt.format(
            summary_statements="\n".join(summary_statements),
            original_statements="\n".join(original_statements)
        )
        
        response = self._call_ollama(prompt)
        
        # Extraire les labels des déclarations à partir de la réponse
        labels = {"TP": [], "FP": [], "FN": []}
        for line in response.split('\n'):
            line = line.strip()
            if "VERDICT: TP" in line or ' TP' in line:
                labels["TP"].append(line)
            elif "VERDICT: FP" in line or ' FP' in line:
                labels["FP"].append(line)
            elif "VERDICT: FN" in line or ' FN' in line:
                labels["FN"].append(line)
        
        return labels
    
    def _direct_verification(self, summary_statements: List[str], text: str) -> Dict[str, List[str]]:
        """
        Vérifie directement les déclarations du résumé par rapport au texte source.
        
        Utilisé par la méthode StSum_Text pour comparer les déclarations du résumé
        directement avec le texte source complet.
        
        Args:
            summary_statements: Liste de déclarations extraites du résumé
            text: Texte source complet
            
        Returns:
            Dictionnaire de classification des déclarations en TP, FP, et FN
        """
        prompt = """
Compare the following statements from the summary with the statements from the original text. Do not add any ponctuation or words. Do not justify your answer.
Label each summary statement as:
- TP (True Positive): If the statement appears in the summary and is directly supported by a statement from the original text.
- FP (False Positive): If the statement appears in the summary but is not directly supported by a statement from the original text.

As you can see in the example bellow, first you have to concatenate the summary statements and the original text statements. 
Then you have to label each statement as TP, FP or FN. Format as follow: VERDICT: TP, VERDICT: FP, VERDICT: FN

Example: 
Summary Statements:
"Albert Einstein was born in Germany",
"Albert Einstein was born in 1879"

Original Text Statements:
"Albert Einstein was born in Spain, in the city of Barcelona in 1879"


Labels:

Albert Einstein was born in 1879. VERDICT: TP
Albert Einstein was born in Germany. VERDICT: FP

END Example



Summary Statements:
{summary_statements}

Original Text Statements or full Text:
{original_statements}

Labels:
""".format(
            summary_statements="\n".join(summary_statements),
            original_statements=text
        )
        
        response = self._call_ollama(prompt)
        
        # Extraire les labels des déclarations
        labels = {"TP": [], "FP": [], "FN": []}
        for line in response.split('\n'):
            line = line.strip()
            if "VERDICT: TP" in line or ' TP' in line:
                labels["TP"].append(line)
            elif "VERDICT: FP" in line or ' FP' in line:
                labels["FP"].append(line)
            elif "VERDICT: FN" in line or ' FN' in line:
                labels["FN"].append(line)
        
        return labels
    
    def _calculate_metrics(self, labels: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calcule les métriques d'évaluation à partir des déclarations labelisées.
        
        Args:
            labels: Classification des déclarations en TP, FP, et FN
            
        Returns:
            Dictionnaire des métriques calculées
        """
        tp = len(labels["TP"])
        fp = len(labels["FP"])
        fn = len(labels["FN"])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_count": tp,
            "fp_count": fp,
            "fn_count": fn
        }
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores SEval-Ex entre les références et les candidats.
        
        Args:
            references: Liste de textes de référence
            candidates: Liste de textes candidats
            **kwargs: Paramètres supplémentaires
                - debug_sample: Nombre d'exemples à traiter en mode debug (défaut: tous)
                - sources: Sources alternatives à utiliser à la place des références
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
        sources = kwargs.get('sources', None)
        use_sources = kwargs.get('use_sources', False)
        
        # Déterminer quels textes utiliser comme références
        if use_sources and sources is not None:
            if len(sources) != len(candidates):
                raise ValueError(f"Le nombre de sources ({len(sources)}) ne correspond pas "
                                f"au nombre de candidats ({len(candidates)})")
            eval_references = sources
            if self.debug_mode:
                logger.debug("Utilisation des sources au lieu des références")
        else:
            eval_references = references
        
        # Limiter le nombre d'exemples en mode debug si spécifié
        if debug_sample is not None and 0 < debug_sample < len(candidates):
            if self.debug_mode:
                logger.debug(f"Mode debug: traitement de {debug_sample} exemples sur {len(candidates)}")
            candidates = candidates[:debug_sample]
            eval_references = eval_references[:debug_sample]
        
        # Initialiser les listes de résultats
        all_metrics = []
        all_statements = []
        all_labels = []
        errors = []
        
        # Traiter chaque paire (référence, candidat)
        for i, (reference, candidate) in enumerate(zip(eval_references, candidates)):
            try:
                if self.verbose or self.debug_mode:
                    logger.info(f"Processing example {i+1}/{len(candidates)}")
                
                if self.debug_mode:
                    logger.debug(f"Référence {i+1}: {reference[:100]}...")
                    logger.debug(f"Candidat {i+1}: {candidate[:100]}...")
                
                # Extraire les déclarations du résumé candidat
                try:
                    candidate_statements = self._extract_statements(candidate)
                    if self.debug_mode:
                        logger.debug(f"Déclarations extraites du candidat {i+1}: {candidate_statements}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction des déclarations du candidat {i+1}: {e}")
                    if self.debug_mode:
                        logger.exception(e)
                    raise
                
                # Traiter selon la méthode choisie
                try:
                    if self.method == "StSum_Text":
                        # Méthode directe: comparer directement les déclarations du résumé avec le texte source
                        if self.debug_mode:
                            logger.debug(f"Utilisation de la méthode StSum_Text pour l'exemple {i+1}")
                        labels = self._direct_verification(candidate_statements, reference)
                    else:
                        # Méthode Base ou Chunked: extraire les déclarations du texte source puis comparer
                        if self.debug_mode:
                            logger.debug(f"Utilisation de la méthode {self.method} pour l'exemple {i+1}")
                        reference_statements = self._extract_statements(reference)
                        if self.debug_mode:
                            logger.debug(f"Déclarations extraites de la référence {i+1}: {reference_statements}")
                        labels = self._verify_statements(candidate_statements, reference_statements)
                except Exception as e:
                    logger.error(f"Erreur lors de la vérification des déclarations pour l'exemple {i+1}: {e}")
                    if self.debug_mode:
                        logger.exception(e)
                    raise
                
                # Calculer les métriques
                metrics = self._calculate_metrics(labels)
                if self.debug_mode:
                    logger.debug(f"Métriques pour l'exemple {i+1}: {metrics}")
                
                all_metrics.append(metrics)
                all_statements.append({"reference": reference, "candidate": candidate, 
                                      "candidate_statements": candidate_statements})
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de l'exemple {i+1}: {e}")
                errors.append({"index": i, "error": str(e), "error_type": type(e).__name__})
                
                # Ajouter des métriques par défaut pour éviter les incohérences
                default_metrics = {"precision": 0, "recall": 0, "f1": 0, "tp_count": 0, "fp_count": 0, "fn_count": 0}
                all_metrics.append(default_metrics)
                all_statements.append({"reference": reference, "candidate": candidate, "candidate_statements": []})
                all_labels.append({"TP": [], "FP": [], "FN": []})
        
        # Calculer les scores moyens
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])
        
        # Convertir les scores individuels en listes pour l'interface TextMetric standard
        individual_precision = [m["precision"] for m in all_metrics]
        individual_recall = [m["recall"] for m in all_metrics]
        individual_f1 = [m["f1"] for m in all_metrics]
        
        # Construire le résultat final
        result = {
            'score': avg_f1,  # F1 comme score principal
            'individual_scores': individual_f1,
            'precision': {
                'score': avg_precision,
                'individual_scores': individual_precision
            },
            'recall': {
                'score': avg_recall,
                'individual_scores': individual_recall
            },
            'f1': {
                'score': avg_f1,
                'individual_scores': individual_f1
            }
        }
        
        # Ajouter les erreurs si en mode debug
        if self.debug_mode and errors:
            result['errors'] = errors
            logger.warning(f"{len(errors)} erreurs rencontrées pendant l'évaluation")
        
        return result