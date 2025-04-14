"""
Module implémentant la métrique MoverScore pour l'évaluation de texte.
"""

from typing import List, Dict, Any, Optional, Set
import os
import sys
import numpy as np
from itertools import chain
from collections import Counter, defaultdict
from math import log
import string
from core.metric_interface import TextMetric

try:
    import torch
    from torch import nn
    from pyemd import emd
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    from pytorch_pretrained_bert.modeling import BertPreTrainedModel
    _MOVERSCORE_AVAILABLE = True
except ImportError:
    _MOVERSCORE_AVAILABLE = False


class MoverScoreMetric(TextMetric):
    """
    Implémentation de la métrique MoverScore pour l'évaluation de texte.
    
    MoverScore utilise la distance Earth Mover's Distance (EMD) entre les embeddings BERT
    des textes pour calculer leur similarité sémantique.
    """
    
    def __init__(self, 
                n_gram: int = 1,
                remove_subwords: bool = True,
                batch_size: int = 48,
                device: str = None,
                model_dir: str = None,
                stopwords_file: str = None):
        """
        Initialise la métrique MoverScore.
        
        Args:
            n_gram: Taille des n-grammes à considérer
            remove_subwords: Supprimer les sous-mots (commençant par ##)
            batch_size: Taille du batch pour l'encodage BERT
            device: Appareil à utiliser ('cuda', 'cpu')
            model_dir: Répertoire contenant le modèle BERT
            stopwords_file: Fichier contenant les mots vides
            
        Raises:
            ImportError: Si les packages requis ne sont pas installés
        """
        if not _MOVERSCORE_AVAILABLE:
            raise ImportError(
                "Les packages 'torch', 'pytorch_pretrained_bert', et 'pyemd' sont requis "
                "pour utiliser MoverScoreMetric. Installez-les avec: "
                "'pip install torch pytorch_pretrained_bert pyemd'."
            )
            
        # Configuration principale
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size
        
        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Charger les mots vides
        self.stopwords = set()
        if stopwords_file is not None and os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(f.read().strip().split(' '))
                
        # Configuration du modèle BERT
        # Définir le répertoire du modèle
        home_dir = os.path.expanduser("~")
        self.model_dir = model_dir or os.path.join(home_dir, '.moverscore')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialiser le modèle et le tokenizer
        self._initialize_bert()
    
    def _initialize_bert(self):
        """Initialise le modèle BERT pour MoverScore."""
        # Définition de la classe BERT pour la classification de séquences
        class BertForSequenceClassification(BertPreTrainedModel):
            def __init__(self, config, num_labels=3):
                super(BertForSequenceClassification, self).__init__(config)
                self.num_labels = num_labels
                self.bert = BertModel(config)
                self.dropout = nn.Dropout(config.hidden_dropout_prob)
                self.classifier = nn.Linear(config.hidden_size, num_labels)
                self.apply(self.init_bert_weights)

            def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=None):
                encoded_layers, pooled_output = self.bert(
                    input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True
                )
                return encoded_layers, pooled_output
        
        # Chargement du tokenizer et du modèle BERT
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir, 3)
        self.model.eval()
        self.model.to(self.device)
        
    def _process_text(self, text: str) -> Set[int]:
        """
        Traite un texte pour l'extraction d'IDF.
        
        Args:
            text: Texte à traiter
            
        Returns:
            Ensemble d'IDs de tokens
        """
        # Tokenisation avec BERT
        tokens = ["[CLS]"] + self.tokenizer.tokenize(text)[:510] + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return set(token_ids)
        
    def _get_idf_dict(self, texts: List[str], nthreads: int = 4) -> Dict[int, float]:
        """
        Calcule le dictionnaire IDF pour une liste de textes.
        
        Args:
            texts: Liste de textes
            nthreads: Nombre de threads pour le traitement parallèle
            
        Returns:
            Dictionnaire IDF {token_id: score_idf}
        """
        # Compter les occurrences de tokens dans tous les documents
        idf_count = Counter()
        num_docs = len(texts)
        
        # Traiter chaque texte pour obtenir ses tokens
        for text in texts:
            idf_count.update(self._process_text(text))
            
        # Calculer les scores IDF
        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
        
        return idf_dict
    
    def _padding(self, arr, pad_token, dtype=torch.long):
        """
        Remplit les séquences pour avoir la même longueur.
        
        Args:
            arr: Liste de séquences
            pad_token: Token de remplissage
            dtype: Type de données pour le tenseur
            
        Returns:
            (padded, lens, mask) - Séquences remplies, longueurs et masques
        """
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
            
        return padded, lens, mask
    
    def _bert_encode(self, texts, attention_mask):
        """
        Encode les textes avec BERT.
        
        Args:
            texts: Textes tokenisés et numérisés
            attention_mask: Masque d'attention
            
        Returns:
            Layers encodées par BERT
        """
        x_seg = torch.zeros_like(texts, dtype=torch.long)
        with torch.no_grad():
            x_encoded_layers, _ = self.model(
                texts, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True
            )
        return x_encoded_layers
    
    def _get_bert_embedding(self, texts, idf_dict):
        """
        Obtient les embeddings BERT pour une liste de textes.
        
        Args:
            texts: Liste de textes
            idf_dict: Dictionnaire IDF
            
        Returns:
            (total_embedding, lens, mask, padded_idf, tokens)
        """
        # Tokeniser et numéraliser les textes
        tokens = [["[CLS]"] + self.tokenizer.tokenize(text)[:510] + ["[SEP]"] for text in texts]
        arr = [self.tokenizer.convert_tokens_to_ids(a) for a in tokens]
        
        # Calculer les poids IDF
        idf_weights = [[idf_dict[i] for i in a] for a in arr]
        
        # Padding
        pad_token = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        padded, lens, mask = self._padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self._padding(idf_weights, pad_token, dtype=torch.float)
        
        # Transférer vers le GPU si disponible
        padded = padded.to(device=self.device)
        mask = mask.to(device=self.device)
        lens = lens.to(device=self.device)
        
        # Obtenir les embeddings BERT
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_embedding = self._bert_encode(
                padded[i:i+self.batch_size], 
                attention_mask=mask[i:i+self.batch_size]
            )
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            
        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens
    
    def _masked_reduce_min(self, tensor, mask):
        """Calcule le minimum masqué."""
        return torch.min(tensor + (1.0 - mask).unsqueeze(-1) * 1e30, dim=1, out=None)
    
    def _masked_reduce_max(self, tensor, mask):
        """Calcule le maximum masqué."""
        return torch.max(tensor - (1.0 - mask).unsqueeze(-1) * 1e30, dim=1, out=None)
    
    def _masked_reduce_mean(self, tensor, mask):
        """Calcule la moyenne masquée."""
        return (tensor * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-10)
    
    def _pairwise_distances(self, x, y=None):
        """
        Calcule les distances euclidiennes entre deux ensembles de vecteurs.
        
        Args:
            x: Premier ensemble de vecteurs
            y: Deuxième ensemble de vecteurs (si None, y=x)
            
        Returns:
            Matrice de distances euclidiennes
        """
        if y is None:
            y = x
            
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        y_t = torch.transpose(y, 0, 1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        
        return torch.clamp(dist, 0.0, np.inf)
    
    def _load_ngram(self, ids, embedding, idf, n, o=1):
        """
        Charge les embeddings des n-grammes.
        
        Args:
            ids: IDs des tokens
            embedding: Embeddings des tokens
            idf: Poids IDF des tokens
            n: Taille des n-grammes
            o: Pas pour la fenêtre glissante
            
        Returns:
            (new_a, new_idf) - Embeddings et IDFs des n-grammes
        """
        # Fonction de division sécurisée
        def _safe_divide(numerator, denominator):
            return numerator / (denominator + 0.00001)
            
        # Fonction pour créer des fenêtres glissantes
        def slide_window(a, w=3, o=2):
            if a.size - w + 1 <= 0:
                w = a.size
            sh = (a.size - w + 1, w)
            st = a.strides * 2
            view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
            return view.copy().tolist()
            
        new_a = []
        new_idf = []

        # Créer des fenêtres glissantes sur les IDs
        slide_wins = slide_window(np.array(ids), w=n, o=o)
        for slide_win in slide_wins:
            new_idf.append(idf[slide_win].sum().item())
            # Pondérer les embeddings par IDF
            scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1).to(self.device)
            tmp = (scale * embedding[slide_win]).sum(0)
            new_a.append(tmp)
            
        new_a = torch.stack(new_a, 0).to(self.device)
        return new_a, new_idf
    
    def word_mover_score(self, refs: List[str], hyps: List[str]) -> List[float]:
        """
        Calcule le score MoverScore entre des références et des hypothèses.
        
        Args:
            refs: Liste de textes de référence
            hyps: Liste de textes d'hypothèse
            
        Returns:
            Liste des scores MoverScore
        """
        # Calculer les dictionnaires IDF
        idf_dict_ref = self._get_idf_dict(refs)
        idf_dict_hyp = self._get_idf_dict(hyps)
        
        # Initialiser les prédictions
        preds = []
        
        # Traiter les textes par lots
        for batch_start in range(0, len(refs), self.batch_size):
            batch_refs = refs[batch_start:batch_start + self.batch_size]
            batch_hyps = hyps[batch_start:batch_start + self.batch_size]
            
            # Obtenir les embeddings BERT
            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self._get_bert_embedding(
                batch_refs, idf_dict_ref
            )
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self._get_bert_embedding(
                batch_hyps, idf_dict_hyp
            )
            
            # Normaliser les embeddings
            ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
            hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
            
            # Extraire les caractéristiques des dernières couches
            ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)
            
            ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
            hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)
            
            ref_embedding_avg = ref_embedding[-5:].mean(0)
            hyp_embedding_avg = hyp_embedding[-5:].mean(0)
            
            # Concaténer les caractéristiques
            ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
            hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)
            
            # Calculer les scores pour chaque paire
            for i in range(len(ref_tokens)):
                # Filtrer les tokens non souhaités
                if self.remove_subwords:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                               w not in set(string.punctuation) and '##' not in w and w not in self.stopwords]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                               w not in set(string.punctuation) and '##' not in w and w not in self.stopwords]
                else:
                    ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                               w not in set(string.punctuation) and w not in self.stopwords]
                    hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                               w not in set(string.punctuation) and w not in self.stopwords]
                
                # Charger les embeddings des n-grammes
                ref_embedding_i, ref_idf_i = self._load_ngram(
                    ref_ids, ref_embedding[i], ref_idf[i], self.n_gram, 1
                )
                hyp_embedding_i, hyp_idf_i = self._load_ngram(
                    hyp_ids, hyp_embedding[i], hyp_idf[i], self.n_gram, 1
                )
                
                # Concaténer et normaliser
                raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)
                
                # Calculer la matrice de distances
                distance_matrix = self._pairwise_distances(raw, raw)
                
                # Préparer les vecteurs de distribution
                c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
                
                c1[:len(ref_idf_i)] = ref_idf_i
                c2[-len(hyp_idf_i):] = hyp_idf_i
                
                # Normaliser les distributions
                c1 = c1 / (np.sum(c1) + 0.00001)
                c2 = c2 / (np.sum(c2) + 0.00001)
                
                # Calculer Earth Mover's Distance
                score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
                preds.append(score)
                
        return preds
    
    @property
    def name(self) -> str:
        """Renvoie le nom de la métrique."""
        return "moverscore"
    
    def compute(self, references: List[str], candidates: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calcule les scores MoverScore entre les références et les candidats.
        
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
        
        # Calculer les scores MoverScore
        individual_scores = self.word_mover_score(references, candidates)
        
        # Calculer le score global (moyenne)
        global_score = np.mean(individual_scores)
        
        return {
            'score': global_score,
            'individual_scores': individual_scores,
            'n_gram': self.n_gram,
            'remove_subwords': self.remove_subwords
        }
