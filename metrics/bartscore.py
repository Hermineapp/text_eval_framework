"""
Module implémentant la métrique BartScore pour l'évaluation de texte.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.metric_interface import TextMetric

try:
    import torch
    import torch.nn as nn
    from transformers import BartTokenizer, BartForConditionalGeneration
    _BARTSCORE_AVAILABLE = True
except ImportError:
    _BARTSCORE_AVAILABLE = False


class BartScorer:
    """
    Class for calculating BartScore between text pairs.
    
    BartScore uses BART's conditional probabilities to evaluate text quality.
    """
    
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path):
        """ Load model from a checkpoint """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError as e:
                print(f'Error scoring batch: {e}')
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                # Continue with other batches instead of exiting
                score_list += [-100.0] * len(src_list)  # Add placeholder scores
        
        return score_list


class BartScoreMetric(TextMetric):
    """
    Implémentation de la métrique BartScore pour l'évaluation de texte.
    
    BartScore utilise les probabilités conditionnelles de BART pour évaluer
    la qualité des textes. Il peut être utilisé en mode source->cible (prec),
    cible->source (recall) ou les deux (F1).
    """
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', 
                direction: str = 'src2tgt', batch_size: int = 4,
                checkpoint_path: Optional[str] = None, 
                device: Optional[str] = None,
                max_length: int = 1024):
        """
        Initialise la métrique BartScore.
        
        Args:
            model_name: Nom du modèle BART à utiliser
            direction: Direction d'évaluation ('src2tgt', 'tgt2src', ou 'avg')
            batch_size: Taille des batchs pour le calcul
            checkpoint_path: Chemin vers un checkpoint pré-entraîné (optionnel)
            device: Appareil à utiliser ('cuda', 'cpu')
            max_length: Longueur maximale des séquences
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _BARTSCORE_AVAILABLE:
            raise ImportError(
                "Les packages 'torch' et 'transformers' sont requis pour utiliser BartScoreMetric. "
                "Installez-les avec 'pip install torch transformers'."
            )
        
        # Valider les paramètres
        if direction not in ['src2tgt', 'tgt2src', 'avg']:
            raise ValueError(
                "Le paramètre 'direction' doit être l'un des suivants: 'src2tgt', 'tgt2src', 'avg'"
            )
        
        self.model_name = model_name
        self.direction = direction
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialiser le scorer
        self.scorer = BartScorer(
            device=self.device,
            max_length=max_length,
            checkpoint=model_name
        )
        
        # Charger un checkpoint personnalisé si spécifié
        if checkpoint_path:
            self.scorer.load(checkpoint_path)
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return f"bartscore_{self.direction}"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores BartScore entre les références et les candidats.
        
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
        
        # Calcul des scores selon la direction
        src2tgt_scores = None
        tgt2src_scores = None
        
        if self.direction in ['src2tgt', 'avg']:
            # Référence -> Candidat (précision)
            src2tgt_scores = self.scorer.score(references, candidates, self.batch_size)
            
        if self.direction in ['tgt2src', 'avg']:
            # Candidat -> Référence (rappel)
            tgt2src_scores = self.scorer.score(candidates, references, self.batch_size)
        
        # Calculer les scores finaux
        if self.direction == 'src2tgt':
            individual_scores = src2tgt_scores
        elif self.direction == 'tgt2src':
            individual_scores = tgt2src_scores
        else:  # 'avg'
            individual_scores = [(s + t) / 2 for s, t in zip(src2tgt_scores, tgt2src_scores)]
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'direction': self.direction,
            'model': self.model_name
        }