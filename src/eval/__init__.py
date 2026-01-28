"""
Evaluation metrics for Visual Storytelling

This module provides comprehensive evaluation metrics for visual storytelling tasks,
including BLEU, CIDEr, ROUGE, BERTScore, and custom storytelling metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from collections import defaultdict
import re
from dataclasses import dataclass

# Import evaluation libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. BLEU and METEOR metrics will be disabled.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available. ROUGE metrics will be disabled.")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.warning("bert-score not available. BERTScore metrics will be disabled.")

try:
    from sacrebleu import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("sacrebleu not available. Some BLEU variants will be disabled.")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Text preprocessing
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    
    # BLEU configuration
    bleu_weights: tuple = (0.25, 0.25, 0.25, 0.25)
    bleu_smoothing: bool = True
    
    # ROUGE configuration
    rouge_metrics: List[str] = None
    
    # BERTScore configuration
    bert_score_model: str = "bert-base-uncased"
    bert_score_lang: str = "en"
    
    # Custom metrics
    compute_narrative_coherence: bool = True
    compute_visual_relevance: bool = True
    
    def __post_init__(self):
        if self.rouge_metrics is None:
            self.rouge_metrics = ["rouge1", "rouge2", "rougeL"]


class StorytellingEvaluator:
    """Comprehensive evaluator for visual storytelling tasks."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self._setup_nltk()
        self._setup_rouge()
        
        # Cache for BERTScore
        self._bert_score_cache = {}
        
        logging.info("StorytellingEvaluator initialized")
    
    def _setup_nltk(self):
        """Setup NLTK components."""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.smoothing_function = SmoothingFunction().method1
            except Exception as e:
                logging.warning(f"Failed to setup NLTK: {e}")
                self.smoothing_function = None
        else:
            self.smoothing_function = None
    
    def _setup_rouge(self):
        """Setup ROUGE scorer."""
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.config.rouge_metrics,
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation.
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        if self.config.remove_stopwords and NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                words = text.split()
                text = ' '.join([word for word in words if word not in stop_words])
            except Exception as e:
                logging.warning(f"Failed to remove stopwords: {e}")
        
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(self.preprocess_text(text))
            except Exception as e:
                logging.warning(f"NLTK tokenization failed: {e}")
        
        # Fallback to simple tokenization
        return self.preprocess_text(text).split()
    
    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute BLEU scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            
        Returns:
            Dict containing BLEU scores
        """
        if not NLTK_AVAILABLE:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        scores = defaultdict(list)
        
        for pred, refs in zip(predictions, references):
            pred_tokens = self.tokenize_text(pred)
            ref_tokens_list = [self.tokenize_text(ref) for ref in refs]
            
            # Individual BLEU scores
            for n in range(1, 5):
                if len(pred_tokens) >= n:
                    weights = [0] * 4
                    weights[n-1] = 1.0
                    
                    if self.smoothing_function:
                        bleu_score = sentence_bleu(
                            ref_tokens_list, pred_tokens,
                            weights=weights,
                            smoothing_function=self.smoothing_function
                        )
                    else:
                        bleu_score = sentence_bleu(
                            ref_tokens_list, pred_tokens,
                            weights=weights
                        )
                    
                    scores[f"bleu_{n}"].append(bleu_score)
                else:
                    scores[f"bleu_{n}"].append(0.0)
            
            # Overall BLEU score
            if self.smoothing_function:
                bleu_score = sentence_bleu(
                    ref_tokens_list, pred_tokens,
                    weights=self.config.bleu_weights,
                    smoothing_function=self.smoothing_function
                )
            else:
                bleu_score = sentence_bleu(
                    ref_tokens_list, pred_tokens,
                    weights=self.config.bleu_weights
                )
            
            scores["bleu"].append(bleu_score)
        
        # Average scores
        return {key: np.mean(values) for key, values in scores.items()}
    
    def compute_rouge(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            
        Returns:
            Dict containing ROUGE scores
        """
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scores = defaultdict(list)
        
        for pred, refs in zip(predictions, references):
            # Use the first reference for ROUGE computation
            ref = refs[0] if refs else ""
            
            rouge_scores = self.rouge_scorer.score(ref, pred)
            
            for metric in self.config.rouge_metrics:
                scores[metric].append(rouge_scores[metric].fmeasure)
        
        return {key: np.mean(values) for key, values in scores.items()}
    
    def compute_meteor(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute METEOR scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            
        Returns:
            Dict containing METEOR scores
        """
        if not NLTK_AVAILABLE:
            return {"meteor": 0.0}
        
        scores = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = self.tokenize_text(pred)
            ref_tokens_list = [self.tokenize_text(ref) for ref in refs]
            
            try:
                meteor_score_val = meteor_score(ref_tokens_list, pred_tokens)
                scores.append(meteor_score_val)
            except Exception as e:
                logging.warning(f"METEOR computation failed: {e}")
                scores.append(0.0)
        
        return {"meteor": np.mean(scores)}
    
    def compute_bert_score(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute BERTScore.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            
        Returns:
            Dict containing BERTScore metrics
        """
        if not BERT_SCORE_AVAILABLE:
            return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}
        
        # Flatten references for BERTScore
        refs_flat = [refs[0] if refs else "" for refs in references]
        
        try:
            P, R, F1 = bert_score(
                predictions, refs_flat,
                model_type=self.config.bert_score_model,
                lang=self.config.bert_score_lang,
                verbose=False
            )
            
            return {
                "bert_score_precision": P.mean().item(),
                "bert_score_recall": R.mean().item(),
                "bert_score_f1": F1.mean().item()
            }
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}
    
    def compute_narrative_coherence(self, predictions: List[str]) -> Dict[str, float]:
        """Compute narrative coherence metrics.
        
        Args:
            predictions: List of predicted texts
            
        Returns:
            Dict containing coherence metrics
        """
        coherence_scores = []
        
        for pred in predictions:
            tokens = self.tokenize_text(pred)
            
            # Simple coherence metrics
            # 1. Average sentence length
            sentences = re.split(r'[.!?]+', pred)
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # 2. Lexical diversity (unique words / total words)
            unique_words = len(set(tokens))
            total_words = len(tokens)
            lexical_diversity = unique_words / total_words if total_words > 0 else 0
            
            # 3. Transition word usage
            transition_words = ['then', 'next', 'after', 'while', 'meanwhile', 'suddenly', 'finally']
            transition_count = sum(1 for word in tokens if word.lower() in transition_words)
            transition_ratio = transition_count / total_words if total_words > 0 else 0
            
            # Combined coherence score
            coherence_score = (avg_sentence_length / 20.0 + lexical_diversity + transition_ratio) / 3.0
            coherence_scores.append(min(coherence_score, 1.0))
        
        return {
            "narrative_coherence": np.mean(coherence_scores),
            "avg_sentence_length": np.mean([len(re.split(r'[.!?]+', pred)) for pred in predictions]),
            "lexical_diversity": np.mean([len(set(self.tokenize_text(pred))) / len(self.tokenize_text(pred)) 
                                        for pred in predictions if len(self.tokenize_text(pred)) > 0])
        }
    
    def compute_visual_relevance(self, predictions: List[str], captions: List[str]) -> Dict[str, float]:
        """Compute visual relevance metrics.
        
        Args:
            predictions: List of predicted texts
            captions: List of image captions
            
        Returns:
            Dict containing relevance metrics
        """
        relevance_scores = []
        
        for pred, caption in zip(predictions, captions):
            pred_tokens = set(self.tokenize_text(pred))
            caption_tokens = set(self.tokenize_text(caption))
            
            # Jaccard similarity
            intersection = len(pred_tokens.intersection(caption_tokens))
            union = len(pred_tokens.union(caption_tokens))
            jaccard_similarity = intersection / union if union > 0 else 0
            
            relevance_scores.append(jaccard_similarity)
        
        return {
            "visual_relevance": np.mean(relevance_scores),
            "caption_overlap": np.mean(relevance_scores)
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[List[str]],
        captions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Comprehensive evaluation of predictions.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            captions: Optional list of image captions
            
        Returns:
            Dict containing all evaluation metrics
        """
        results = {}
        
        # Standard metrics
        results.update(self.compute_bleu(predictions, references))
        results.update(self.compute_rouge(predictions, references))
        results.update(self.compute_meteor(predictions, references))
        results.update(self.compute_bert_score(predictions, references))
        
        # Custom storytelling metrics
        if self.config.compute_narrative_coherence:
            results.update(self.compute_narrative_coherence(predictions))
        
        if self.config.compute_visual_relevance and captions:
            results.update(self.compute_visual_relevance(predictions, captions))
        
        # Overall score (weighted combination)
        weights = {
            'bleu': 0.2,
            'rouge1': 0.2,
            'rougeL': 0.2,
            'meteor': 0.2,
            'bert_score_f1': 0.2
        }
        
        overall_score = sum(results.get(metric, 0) * weight for metric, weight in weights.items())
        results['overall_score'] = overall_score
        
        return results
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results in a formatted way."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Group metrics
        metric_groups = {
            "BLEU Scores": ["bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4"],
            "ROUGE Scores": ["rouge1", "rouge2", "rougeL"],
            "Other Metrics": ["meteor", "bert_score_f1", "bert_score_precision", "bert_score_recall"],
            "Storytelling Metrics": ["narrative_coherence", "visual_relevance", "lexical_diversity"],
            "Overall": ["overall_score"]
        }
        
        for group_name, metrics in metric_groups.items():
            print(f"\n{group_name}:")
            print("-" * len(group_name))
            for metric in metrics:
                if metric in results:
                    print(f"  {metric:25}: {results[metric]:.4f}")
        
        print("\n" + "="*50)
