"""
SQuAD v2.0 Task Handler
Handles prompt generation, answer extraction, and F1/Exact Match evaluation
"""

import re
import json
import string
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SQuADTask:
    """SQuAD v2.0 extractive QA task"""
    
    # Few-shot examples for consistent prompting
    FEW_SHOT_EXAMPLES = [
        {
            "context": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth and wood, built along the historical northern borders of China to protect the Chinese states and empires against the raids and invasions of various nomadic groups.",
            "question": "What was the Great Wall of China built with?",
            "answer": "stone, brick, tamped earth and wood"
        },
        {
            "context": "Elizabeth II was Queen of the United Kingdom and the Commonwealth realms from February 1952 until her death on September 8, 2022.",
            "question": "When did Queen Elizabeth II die?",
            "answer": "September 8, 2022"
        }
    ]
    
    def __init__(self):
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_tokens = 64
    
    def create_prompt(self, context: str, question: str, include_fewshot: bool = True) -> str:
        """
        Create a prompt for SQuAD QA task
        
        Args:
            context: The passage to extract answer from
            question: The question to answer
            include_fewshot: Whether to include few-shot examples
        
        Returns:
            Complete prompt string
        """
        prompt = ""
        
        # Add few-shot examples
        if include_fewshot:
            for ex in self.FEW_SHOT_EXAMPLES:
                prompt += f"Context: {ex['context']}\n"
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Answer: {ex['answer']}\n\n"
        
        # Add actual task
        prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        prompt += f"Answer: "
        
        return prompt
    
    def extract_answer(self, generated_text: str) -> str:
        """
        Extract answer from model output
        
        Args:
            generated_text: Raw model output
        
        Returns:
            Extracted answer (first sentence or until punctuation)
        """
        # Remove leading/trailing whitespace
        text = generated_text.strip()
        
        # Stop at common punctuation
        for punct in ['.', '\n', '?']:
            if punct in text:
                text = text[:text.index(punct)].strip()
        
        # Limit to reasonable length (max 100 chars for SQuAD answers)
        if len(text) > 100:
            text = text[:100]
        
        return text
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for fair comparison
        (from official SQuAD evaluation script)
        """
        def remove_articles(s):
            return re.sub(r'\b(a|an|the)\b', ' ', s)
        
        def white_space_fix(s):
            return ' '.join(s.split())
        
        def remove_punc(s):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in s if ch not in exclude)
        
        def lower(s):
            return s.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        """Check exact match (normalized)"""
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)
    
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score between prediction and ground truth
        (from official SQuAD evaluation script)
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        common = set(pred_tokens) & set(truth_tokens)
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """
        Evaluate prediction against multiple ground truth answers
        
        Args:
            prediction: Model's predicted answer
            ground_truths: List of valid answers (SQuAD provides multiple)
        
        Returns:
            Dict with 'exact_match' and 'f1' scores
        """
        exact_match_scores = [
            self.exact_match(prediction, gt) for gt in ground_truths
        ]
        f1_scores = [
            self.compute_f1(prediction, gt) for gt in ground_truths
        ]
        
        return {
            'exact_match': max(exact_match_scores),  # 1 if any match, else 0
            'f1': max(f1_scores)  # Best F1 against any reference
        }
    
    def batch_evaluate(self, predictions: List[str], references: List[Dict]) -> Dict:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of model predictions
            references: List of dicts with 'answers' field (SQuAD format)
        
        Returns:
            Aggregate metrics
        """
        exact_matches = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Handle both SQuAD formats:
            # 1) {"answers": [{"text": "..."} , ...]}
            # 2) {"text": ["...", ...], "answer_start": [...]}
            ground_truths = []
            if isinstance(ref, dict):
                if isinstance(ref.get('answers'), list):
                    for ans in ref['answers']:
                        if isinstance(ans, dict) and isinstance(ans.get('text'), str):
                            ground_truths.append(ans['text'])
                elif isinstance(ref.get('text'), list):
                    ground_truths = [t for t in ref['text'] if isinstance(t, str)]

            if not ground_truths:
                # Graceful fallback to avoid crashing a full run.
                ground_truths = [""]

            scores = self.evaluate(pred, ground_truths)
            exact_matches.append(scores['exact_match'])
            f1_scores.append(scores['f1'])
        
        return {
            'exact_match': sum(exact_matches) / len(exact_matches) if exact_matches else 0.0,
            'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'count': len(predictions)
        }


class StructuredOutputTask(SQuADTask):
    """
    Structured output generation task
    Prompts model to generate JSON responses and validates structure
    """
    
    def create_prompt(self, context: str, question: str) -> str:
        """Create prompt requesting JSON output"""
        prompt = f"""Given the following context, answer the question in JSON format.

Context: {context}

Question: {question}

Respond ONLY with valid JSON in this format:
{{"answer": "<your answer>", "confidence": "high/medium/low"}}

JSON Response:"""
        return prompt

    def extract_answer(self, generated_text: str) -> str:
        """
        Extract answer text from a structured JSON response.
        Falls back to base extraction when JSON parsing fails.
        """
        json_obj = self.extract_json(generated_text)
        if json_obj and isinstance(json_obj.get("answer"), str):
            return json_obj["answer"].strip()
        return super().extract_answer(generated_text)
    
    def extract_json(self, text: str) -> Dict:
        """Extract JSON from model output"""
        # Try to find JSON block
        import json as json_lib
        
        # Look for {...}
        start = text.find('{')
        if start == -1:
            return None
        
        # Find matching closing brace
        brace_count = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if end is None:
            return None
        
        json_str = text[start:end]
        
        try:
            return json_lib.loads(json_str)
        except json_lib.JSONDecodeError:
            return None
    
    def validate_structure(self, json_obj: Dict) -> Tuple[bool, str]:
        """
        Validate JSON structure
        
        Returns:
            (is_valid, error_message)
        """
        if json_obj is None:
            return False, "Invalid JSON"
        
        required_fields = {'answer', 'confidence'}
        missing = required_fields - set(json_obj.keys())
        
        if missing:
            return False, f"Missing fields: {missing}"
        
        valid_confidences = {'high', 'medium', 'low'}
        if json_obj['confidence'].lower() not in valid_confidences:
            return False, f"Invalid confidence: {json_obj['confidence']}"
        
        if not isinstance(json_obj['answer'], str):
            return False, "Answer must be string"
        
        return True, ""

    def batch_evaluate(self, predictions: List[str], references: List[Dict]) -> Dict:
        """Same QA scoring as SQuAD; predictions are plain answer strings after extract_answer."""
        return super().batch_evaluate(predictions, references)


class LongContextTask(SQuADTask):
    """
    Long-context summarization task
    Tests model ability to handle long prompts (2k-4k tokens)
    """
    
    def create_prompt(self, passages: List[str], question: str) -> str:
        """
        Create long-context prompt by concatenating passages
        
        Args:
            passages: List of text passages (each ~400-500 tokens)
            question: Question that requires synthesizing across passages
        """
        combined = "\n\n".join(passages)
        
        prompt = f"""Read the following passages and answer the question based on information from them:

{combined}

Question: {question}

Answer: """
        return prompt

    def extract_answer(self, generated_text: str) -> str:
        """Reuse SQuAD span extraction for free-form answer after long-context prompt."""
        return super().extract_answer(generated_text)

    def batch_evaluate(self, predictions: List[str], references: List[Dict]) -> Dict:
        return super().batch_evaluate(predictions, references)


def get_task_handler(task_type: str) -> Any:
    """Factory function to get appropriate task handler"""
    if task_type == "interactive_chat":
        return SQuADTask()
    elif task_type == "structured_output":
        return StructuredOutputTask()
    elif task_type == "long_context":
        return LongContextTask()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
