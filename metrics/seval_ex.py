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
                 verbose: bool = False):
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
        
        # Initialiser le client Ollama
        try:
            self.ollama_client = ollama.Client(host=ollama_base_url)
            logger.info(f"SEval-Ex initialisé avec le modèle {model_name} et la méthode {method}")
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
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Appelle l'API Ollama avec le prompt spécifié.
        
        Args:
            prompt: Le prompt à envoyer à Ollama
            
        Returns:
            La réponse générée par Ollama
        """
        try:
            if self.verbose:
                logger.info(f"Sending prompt to Ollama: {prompt[:100]}...")
            
            completion = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": self.temperature}
            )
            
            if self.verbose:
                logger.info(f"Received response from Ollama: {completion['response'][:100]}...")
            
            return completion['response']
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            # Renvoyer une réponse vide en cas d'erreur
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
        Compare the following statements from the summary with the original text. Do not add any ponctuation or words. Do not justify your answer.
        Label each summary statement as:
        - TP (True Positive): If the statement appears in the summary and is directly supported by the original text.
        - FP (False Positive): If the statement appears in the summary but is not directly supported by the original text.
        - FN (False Negative): If a key information appears in the original text but is missing from the summary statements.
        
        Format as follow: VERDICT: TP, VERDICT: FP, VERDICT: FN
        
        Summary Statements:
        {summary_statements}
        
        Original Text:
        {text}
        
        Labels:
        """.format(
            summary_statements="\n".join(summary_statements),
            text=text
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
            
        Returns:
            Dict contenant le score global et les scores individuels
        """
        if len(references) != len(candidates):
            raise ValueError(
                f"Le nombre de références ({len(references)}) ne correspond pas "
                f"au nombre de candidats ({len(candidates)})"
            )
        
        # Initialiser les listes de résultats
        all_metrics = []
        all_statements = []
        all_labels = []
        
        # Traiter chaque paire (référence, candidat)
        for i, (reference, candidate) in enumerate(zip(references, candidates)):
            if self.verbose:
                logger.info(f"Processing example {i+1}/{len(references)}")
            
            # Extraire les déclarations du résumé candidat
            candidate_statements = self._extract_statements(candidate)
            
            # Traiter selon la méthode choisie
            if self.method == "StSum_Text":
                # Méthode directe: comparer directement les déclarations du résumé avec le texte source
                labels = self._direct_verification(candidate_statements, reference)
            else:
                # Méthode Base ou Chunked: extraire les déclarations du texte source puis comparer
                reference_statements = self._extract_statements(reference)
                labels = self._verify_statements(candidate_statements, reference_statements)
            
            # Calculer les métriques
            metrics = self._calculate_metrics(labels)
            all_metrics.append(metrics)
            all_statements.append({"reference": reference, "candidate": candidate, 
                                  "candidate_statements": candidate_statements})
            all_labels.append(labels)
        
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
            },
            'statement_details': [
                {
                    "statements": all_statements[i],
                    "labels": all_labels[i],
                    "metrics": all_metrics[i]
                }
                for i in range(len(all_metrics))
            ]
        }
        
        return result
