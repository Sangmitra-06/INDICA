"""
LLM-as-a-Judge for Cultural Short Answers
=========================================

Implements an automated evaluation framework for scoring short-answer generation tasks
using a strong LLM as a judge. Assesses factual correctness against gold standard answers
with specific guidelines for handling cultural nuances and partial correctness.
"""

import json
import requests
import re
from typing import Dict, List
from collections import defaultdict
import time
import os
import argparse
from pathlib import Path


class LLMJudgeEvaluator:
    def __init__(self, api_key: str, model: str = "google/gemini-3-flash-preview"):
        """
        Initialize the LLM judge.
        
        Args:
            api_key: API Key for the model provider (OpenRouter).
            model: The judge model identifier (default: google/gemini-3-flash-preview).
        """
        self.api_key = api_key
        self.model = model

        # Define evaluation criteria and judgement protocol
        self.system_prompt = """You are an expert evaluator for cultural knowledge questions about India.

        Your task: Determine if a model's answer is factually correct compared to the gold standard answer.

        Guidelines:
        1. CORRECT: The model's answer conveys ONLY the same core information as the gold answer
           - Paraphrasing is acceptable
           - ALL details in the model's answer must appear in the gold answer
           - Equivalent terms are acceptable (e.g., "clay stove" = "chulha")
           - The answer should not contain significant additional facts, examples, or details beyond what the gold answer provides

        2. PARTIALLY_CORRECT: The model's answer has some correct information from the gold answer but:
           - Misses details from the gold answer
           - Contains additional information, facts, or examples that are not present in the gold answer (even if factually correct)
           - Provides extra context or details that go beyond the scope of the gold answer
           - Is too vague or incomplete

        3. INCORRECT: The model's answer:
           - Contradicts the gold answer with wrong facts
           - Provides completely different information
           - Misses the main point entirely

        Important:
        - Focus on factual accuracy, not writing style
        - Consider cultural context and regional variations
        - Be strict about factual contradictions (e.g., "jewelry" ≠ "cash")
        - If the model adds information not in the gold answer (like additional examples, regional variations, or extra details), mark as PARTIALLY_CORRECT even if the added information is accurate
        - The gold answer defines the scope - answers should not exceed that scope

        Output format: JSON with fields:
        {
          "label": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT",
          "reasoning": "brief explanation",
          "key_discrepancies": ["list any factual errors or significant additions"]
        }"""

    def clean_json_response(self, response_text: str) -> str:
        """
        Sanitize malformed JSON responses from the model.
        Handles common LLM output issues like markdown blocks, trailing text, and invalid control characters.
        """
        # Remove any text before first {
        start = response_text.find('{')
        if start > 0:
            response_text = response_text[start:]

        # Remove any text after last }
        end = response_text.rfind('}')
        if end > 0:
            response_text = response_text[:end + 1]

        # Replace invalid control characters (but preserve newlines and tabs in strings)
        response_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', response_text)

        return response_text

    def extract_label_from_text(self, text: str) -> str:
        """
        Fallback mechanism to parse evaluation labels from unstructured text
        when JSON parsing fails completely.
        """
        text_upper = text.upper()

        # Look for keywords (order matters - check most specific first)
        if 'PARTIALLY_CORRECT' in text_upper or 'PARTIALLY CORRECT' in text_upper or 'PARTIAL' in text_upper:
            return 'partially_correct'
        elif 'INCORRECT' in text_upper:
            return 'incorrect'
        elif 'CORRECT' in text_upper:
            return 'correct'

        # Default
        return 'error'

    def evaluate_single_answer(
            self,
            question: str,
            gold_answer: str,
            predicted_answer: str,
            max_retries: int = 3
    ) -> Dict:
        """
        Evaluate a single instance using the judge model.
        Includes robust error handling and retry logic for API interactions.
        
        Returns:
            Dictionary containing the verification label, reasoning, and cost metrics.
        """
        user_prompt = f"""Question: {question}

Gold Standard Answer: {gold_answer}

Model's Answer: {predicted_answer}

Evaluate the model's answer."""

        for attempt in range(max_retries):
            try:
                # Use OpenRouter API for Gemini
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                }

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json=payload,
                    timeout=60  # Increased timeout
                )

                if response.status_code != 200:
                    if attempt == max_retries - 1:
                        return self._fallback_judgment(f"API Error {response.status_code}")
                    time.sleep(2 ** attempt)
                    continue

                data = response.json()

                # Handle case where API returns a list instead of dict
                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        if attempt == max_retries - 1:
                            return self._fallback_judgment("Empty response list")
                        time.sleep(1)
                        continue

                if 'choices' not in data or not data['choices']:
                    if attempt == max_retries - 1:
                        return self._fallback_judgment("No choices in response")
                    time.sleep(1)
                    continue

                raw_content = data['choices'][0]['message']['content']

                # Try to parse JSON
                try:
                    judgment = json.loads(raw_content)

                except json.JSONDecodeError as json_err:
                    # Try to clean and parse again
                    try:
                        cleaned = self.clean_json_response(raw_content)
                        judgment = json.loads(cleaned)

                    except json.JSONDecodeError:
                        # Last resort: extract label from text
                        if attempt == max_retries - 1:
                            label = self.extract_label_from_text(raw_content)
                            return {
                                'label': label,
                                'reasoning': f'Extracted from malformed JSON. Raw response: {raw_content[:300]}...',
                                'key_discrepancies': [],
                                'model_used': self.model,
                                'tokens_used': data.get('usage', {}).get('total_tokens', 0),
                                'cost_usd': self._estimate_cost(data.get('usage', {})),
                                'parsing_fallback': True
                            }
                        # Retry with exponential backoff
                        time.sleep(2 ** attempt)
                        continue

                # Normalize label
                label = judgment.get('label', 'INCORRECT').upper().replace(' ', '_')
                if label not in ['CORRECT', 'PARTIALLY_CORRECT', 'INCORRECT']:
                    # Try to fix common variations
                    if 'PARTIAL' in label:
                        label = 'PARTIALLY_CORRECT'
                    else:
                        label = 'INCORRECT'

                usage = data.get('usage', {})

                return {
                    'label': label.lower(),
                    'reasoning': str(judgment.get('reasoning', ''))[:1000],  # Truncate very long reasoning
                    'key_discrepancies': judgment.get('key_discrepancies', [])[:10],  # Limit list size
                    'model_used': self.model,
                    'tokens_used': usage.get('total_tokens', 0),
                    'cost_usd': self._estimate_cost(usage)
                }

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return self._fallback_judgment("Request timeout")
                time.sleep(2 ** attempt)

            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries - 1:
                    return self._fallback_judgment(f"Connection error: {str(e)[:100]}")
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return self._fallback_judgment(f"Request error: {str(e)[:100]}")
                time.sleep(2 ** attempt)

            except Exception as e:
                if attempt == max_retries - 1:
                    return self._fallback_judgment(f"Unexpected error: {str(e)[:100]}")
                time.sleep(2 ** attempt)

        return self._fallback_judgment("Max retries exceeded")

    def _estimate_cost(self, usage) -> float:
        """Estimate cost based on token usage."""
        if not usage:
            return 0.0

        # Pricing for Gemini Flash 3.0
        if "gemini-flash-3.0" in self.model or "gemini-3-flash" in self.model:
            input_cost = usage.get('prompt_tokens', 0) * 0.50 / 1_000_000
            output_cost = usage.get('completion_tokens', 0) * 3.00 / 1_000_000
        else:
            input_cost = usage.get('prompt_tokens', 0) * 0.50 / 1_000_000
            output_cost = usage.get('completion_tokens', 0) * 1.50 / 1_000_000

        return input_cost + output_cost

    def _fallback_judgment(self, error_msg: str = "API call failed") -> Dict:
        """Return fallback judgment when API fails."""
        return {
            'label': 'error',
            'reasoning': error_msg,
            'key_discrepancies': [],
            'model_used': self.model,
            'tokens_used': 0,
            'cost_usd': 0.0,
            'api_error': True
        }

    def evaluate_dataset(
            self,
            results_file: str,
            cache_file: str = None,
            batch_size: int = 10
    ) -> Dict:
        """
        Evaluate all answers from a JSONL file.

        Args:
            results_file: Path to model results JSONL
            cache_file: Optional path to cache judgments (to resume if interrupted)
            batch_size: Print progress every N evaluations

        Returns:
            Dictionary with evaluation results
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f if line.strip()]

        # Load cache if exists
        cache = {}
        if cache_file:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    cache = {
                        f"{r['question_id']}_{r['run_number']}": r
                        for r in cache_data.get('all_evaluations', [])
                    }
                print(f"Loaded {len(cache)} cached evaluations")
            except FileNotFoundError:
                print("No cache file found, starting fresh")

        # Aggregate by question
        by_question = defaultdict(list)
        for result in results:
            q_id = result['question_id']
            by_question[q_id].append(result)

        all_evaluations = []
        question_scores = {}
        total_cost = 0.0
        total_tokens = 0

        print(f"\n{'=' * 80}")
        print(f"Evaluating {len(results)} answers using LLM judge ({self.model})")
        print(f"{'=' * 80}\n")

        eval_count = 0

        for q_id, runs in by_question.items():
            question_evals = []

            for run in runs:
                cache_key = f"{q_id}_{run['run_number']}"

                # Check cache
                if cache_key in cache:
                    eval_result = cache[cache_key]
                    # Don't print every cached entry to reduce spam
                else:
                    # Evaluate
                    question = run.get('question_text', '')
                    gold = run['gold_answer']
                    pred = run.get('extracted_answer') or run.get('parsed_answer')

                    if not pred:
                        continue

                    eval_result = self.evaluate_single_answer(question, gold, pred)
                    eval_result['question_id'] = q_id
                    eval_result['run_number'] = run['run_number']
                    eval_result['question_text'] = question
                    eval_result['gold_answer'] = gold
                    eval_result['predicted_answer'] = pred

                    total_cost += eval_result['cost_usd']
                    total_tokens += eval_result['tokens_used']

                    eval_count += 1

                    if eval_count % batch_size == 0:
                        print(f"  Evaluated {eval_count}/{len(results)} | Cost so far: ${total_cost:.4f}")

                    # Save to cache periodically
                    if cache_file and eval_count % 50 == 0:
                        self._save_cache(cache_file, all_evaluations + [eval_result])

                question_evals.append(eval_result)
                all_evaluations.append(eval_result)

            # Average for this question
            if question_evals:
                correct_count = sum(1 for e in question_evals if e['label'] == 'correct')
                partial_count = sum(1 for e in question_evals if e['label'] == 'partially_correct')

                question_scores[q_id] = {
                    'accuracy': correct_count / len(question_evals),
                    'partial_credit_accuracy': (correct_count + 0.5 * partial_count) / len(question_evals),
                    'num_runs': len(question_evals),
                    'evaluations': question_evals
                }

        # Overall statistics
        total_evals = len(all_evaluations)
        correct = sum(1 for e in all_evaluations if e['label'] == 'correct')
        partial = sum(1 for e in all_evaluations if e['label'] == 'partially_correct')
        incorrect = sum(1 for e in all_evaluations if e['label'] == 'incorrect')
        errors = sum(1 for e in all_evaluations if e['label'] == 'error')

        results_dict = {
            'overall_statistics': {
                'total_evaluations': total_evals,
                'unique_questions': len(question_scores),
                'correct_count': correct,
                'partially_correct_count': partial,
                'incorrect_count': incorrect,
                'error_count': errors,
                'accuracy': correct / total_evals if total_evals > 0 else 0,
                'partial_credit_accuracy': (correct + 0.5 * partial) / total_evals if total_evals > 0 else 0,
                'total_cost_usd': round(total_cost, 4),
                'total_tokens': total_tokens,
                'judge_model': self.model
            },
            'per_question_scores': question_scores,
            'all_evaluations': all_evaluations
        }

        # Final cache save
        if cache_file:
            self._save_cache(cache_file, all_evaluations)

        return results_dict

    def _save_cache(self, cache_file: str, evaluations: List[Dict]):
        """Save evaluations to cache file with UTF-8 encoding."""
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'all_evaluations': evaluations}, f, indent=2, ensure_ascii=False)

    def generate_report(self, evaluation_results: Dict, output_file: str = None):
        """Generate human-readable report."""
        stats = evaluation_results['overall_statistics']

        print("\n" + "=" * 80)
        print("SHORT ANSWER EVALUATION REPORT (LLM-as-Judge)")
        print("=" * 80)

        print(f"\nOVERALL STATISTICS:")
        print(f"   Judge Model: {stats['judge_model']}")
        print(f"   Total Evaluations: {stats['total_evaluations']}")
        print(f"   Unique Questions: {stats['unique_questions']}")
        print(f"   Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"   Total Tokens: {stats['total_tokens']:,}")
        print(f"   ")
        print(
            f"   ✓ Correct: {stats['correct_count']} ({stats['correct_count'] / stats['total_evaluations'] * 100:.1f}%)")
        print(
            f"   ~ Partially Correct: {stats['partially_correct_count']} ({stats['partially_correct_count'] / stats['total_evaluations'] * 100:.1f}%)")
        print(
            f"   ✗ Incorrect: {stats['incorrect_count']} ({stats['incorrect_count'] / stats['total_evaluations'] * 100:.1f}%)")
        if stats['error_count'] > 0:
            print(
                f"   [Errors]: {stats['error_count']} ({stats['error_count'] / stats['total_evaluations'] * 100:.1f}%)")
        print(f"   ")
        print(f"   Accuracy (strict): {stats['accuracy'] * 100:.1f}%")
        print(f"   Accuracy (with partial credit): {stats['partial_credit_accuracy'] * 100:.1f}%")

        # Show examples by category
        all_evals = evaluation_results['all_evaluations']

        # Incorrect examples with reasoning
        incorrect_examples = [e for e in all_evals if e['label'] == 'incorrect'][:3]
        if incorrect_examples:
            print(f"\n[INCORRECT ANSWER EXAMPLES]:")
            for i, eval_result in enumerate(incorrect_examples):
                print(f"\n   Example {i + 1}:")
                print(f"   Question: {eval_result.get('question_text', '')[:80]}...")
                print(f"   Gold: {eval_result['gold_answer'][:80]}...")
                print(f"   Pred: {eval_result['predicted_answer'][:80]}...")
                print(f"   Judge's reasoning: {eval_result['reasoning'][:150]}...")
                if eval_result.get('key_discrepancies'):
                    print(f"   Discrepancies: {eval_result['key_discrepancies'][:2]}")

        # Correct examples
        correct_examples = [e for e in all_evals if e['label'] == 'correct'][:2]
        if correct_examples:
            print(f"\n[CORRECT ANSWER EXAMPLES]:")
            for i, eval_result in enumerate(correct_examples):
                print(f"\n   Example {i + 1}:")
                print(f"   Question: {eval_result.get('question_text', '')[:80]}...")
                print(f"   Gold: {eval_result['gold_answer'][:80]}...")
                print(f"   Pred: {eval_result['predicted_answer'][:80]}...")
                print(f"   Judge's reasoning: {eval_result['reasoning'][:150]}...")

        print("\n" + "=" * 80 + "\n")

        # Save to JSON with UTF-8 encoding
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"Full results saved to: {output_file}\n")


def evaluate_single_model(
        model_name: str,
        results_dir: str,
        output_dir: str,
        api_key: str,
        judge_model: str = "google/gemini-3-flash-preview"
):
    """
    Driver function to process evaluation for a single target model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Construct the expected filename
    # Handle both formats: "model-name" and "provider/model-name"
    model_safe = model_name.replace('/', '_').replace(':', '_')
    results_filename = f"{model_safe}_SHORT_ANSWER_results.jsonl"
    results_file = os.path.join(results_dir, results_filename)

    if not os.path.exists(results_file):
        print(f"[Error] Results file not found: {results_file}")
        print(f"\nAvailable SHORT_ANSWER files in {results_dir}:")
        available = list(Path(results_dir).glob("*_SHORT_ANSWER_results.jsonl"))
        for f in available:
            print(f"   - {f.name}")
        return

    print(f"\n{'=' * 80}")
    print(f"Evaluating Model: {model_name}")
    print(f"Results File: {results_filename}")
    print(f"Judge Model: {judge_model}")
    print(f"{'=' * 80}")

    cache_file = os.path.join(output_dir, f"{model_safe}_cache.json")
    output_file = os.path.join(output_dir, f"{model_safe}_short_answer_evaluation.json")

    evaluator = LLMJudgeEvaluator(api_key=api_key, model=judge_model)

    try:
        results = evaluator.evaluate_dataset(
            results_file,
            cache_file=cache_file
        )

        evaluator.generate_report(results, output_file)

        print(f"\n[Success] Evaluation complete for {model_name}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"[Error] Evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate short answers using LLM-as-Judge")
    parser.add_argument("model",
                        help="Model name to evaluate (e.g., 'gpt-5.2-2025-12-11' or 'google/gemini-3-flash-preview')")
    parser.add_argument("--results-dir", default="evaluation_results/processed",
                        help="Directory containing result files")
    parser.add_argument("--output-dir", default="short_answer_evaluations",
                        help="Directory to save evaluation results")
    parser.add_argument("--judge-model", default="google/gemini-3-flash-preview",
                        help="Model to use as judge")
    parser.add_argument("--api-key",
                        default="your-api-key-here",
                        help="OpenRouter API key")

    args = parser.parse_args()

    evaluate_single_model(
        model_name=args.model,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        judge_model=args.judge_model
    )