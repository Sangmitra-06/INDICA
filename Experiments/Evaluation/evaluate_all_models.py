"""
Model Evaluation Script
=======================

Main driver script for evaluating Language Models on cultural commonsense tasks.
Supports multiple evaluation categories (RA-MCQ, RASA)
and handles prompt generation, model querying, and result logging.

Usage:
    python evaluate_all_models.py "model_name" --category [CATEGORY]
"""

import json
import time
import sys
import os
import argparse
from results_logger import ResultsLogger
from eval_utils import (
    create_ramcq_prompt,
    create_rasa_prompt,
    query_model,
    extract_answer_letter,
    MODEL_PRICING
)

# Configuration
OUTPUT_DIR = "evaluation_results"

# Dataset Paths
DATASETS = {
    "RA_MCQ": "../Evaluation_Questions/mcq_questions.json",
    "RASA": "../Evaluation_Questions/short_answer_questions.json"
}


def resolve_path(path):
    """
    Resolve local resource paths.
    Checks multiple potential locations to support different execution contexts.
    """
    if os.path.exists(path): return path
    script_dir = os.path.dirname(__file__)
    abs_path = os.path.join(script_dir, path)
    if os.path.exists(abs_path): return abs_path

    # Try alternate location
    alt_path = path.replace("../Evaluation_Questions", "Evaluation_Questions")
    if os.path.exists(alt_path): return alt_path

    return path


def load_questions(category, file_path):
    """
    Load and standardize question data structures.
    Adapts different source schemas into a unified list format.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Error] Could not find file {file_path}")
        return []

    if category == "RA_MCQ":
        return data['questions']
    elif category == "RASA":
        questions = []
        for region, q_list in data.items():
            if isinstance(q_list, list):
                questions.extend(q_list)
        return questions
    return []


def evaluate_model(model: str, category: str, runs_per_question: int = 30, temperature: float = 1.0):
    """
    Execute evaluation loop for a specific model and task.
    Iterates through questions, handles retries, and logs detailed results.
    """

    if model not in MODEL_PRICING:
        print(f"[Error] Model '{model}' not found in configuration.")
        return

    config = MODEL_PRICING[model]
    provider = config["provider"]

    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {model}")
    print(f"CATEGORY: {category}")
    print(f"Run Count: {runs_per_question}")
    print(f"Provider: {provider}")
    print(f"{'=' * 60}")

    file_path = resolve_path(DATASETS[category])
    questions = load_questions(category, file_path)

    if not questions:
        print("[Error] No questions loaded. Check file paths.")
        return

    print(f"\nDataset: {len(questions)} questions")

    # Initialize Logger
    logger = ResultsLogger(OUTPUT_DIR, category)

    failed_calls = 0
    total_cost = 0
    start_time = time.time()

    for q_idx, question in enumerate(questions):
        print(f"\n{'─' * 60}")
        print(f"Question {q_idx + 1}/{len(questions)}")

        for run in range(1, runs_per_question + 1):
            if run % 5 == 0 or run == 1:  # Reduce spam for 30 runs
                print(f"  Run {run}/{runs_per_question}...", end=" ")

            # 1. Initialize prompt context
            prompt = ""
            gold_answer = ""
            q_id = "unknown"
            q_text = ""
            current_mapping = "N/A"

            # 2. Generate Shuffled Prompts
            if category == "RA_MCQ":
                # Always shuffle
                prompt, mapping = create_ramcq_prompt(question, shuffle_options=True)
                gold_answer = str(mapping)
                current_mapping = mapping
                q_id = question.get('question_id', 'unknown')
                q_text = question.get('question_text', '')

            elif category == "RASA":
                prompt = create_rasa_prompt(question)
                gold_answer = question.get('answer', '')
                q_id = question.get('question_id', 'unknown')
                q_text = question.get('question', '')

            # 3. Execute Model Query
            response, usage = query_model(prompt, model, provider, temperature)

            metadata = {
                "run": run,
                "usage": usage,
                "mapping": current_mapping,
                "shuffle": True
            }

            if response is None:
                print(" [Failed]")
                failed_calls += 1
                logger.log_result(model, q_id, q_text, gold_answer, None, "TIMEOUT_OR_API_ERROR", metadata)
                time.sleep(2)
                continue

            # 4. Parse
            parsed = "N/A"
            if category == "RASA":
                parsed = response.strip()
                if run % 5 == 0 or run == 1: print(" [Done]")
            else:
                parsed = extract_answer_letter(response)
                if run % 5 == 0 or run == 1:
                    print(f" -> {parsed}")

            # 5. Log Results
            logger.log_result(
                model_name=model,
                question_id=q_id,
                question_text=q_text,
                gold_answer=gold_answer,
                parsed_answer=parsed,
                raw_response=response,
                metadata=metadata
            )

            # Accumulate cost estimates
            if usage:
                cost = ((usage.get('prompt_tokens', 0) / 1_000_000) * config['input'] +
                        (usage.get('completion_tokens', 0) / 1_000_000) * config['output'])
                total_cost += cost

            # Small delay to prevent rate limit aggression on free loops
            time.sleep(0.5)

        # Progress
        elapsed = time.time() - start_time
        avg = elapsed / (q_idx + 1)
        remain = avg * (len(questions) - q_idx - 1)
        print(
            f"  [Progress: {(q_idx + 1) / len(questions) * 100:.1f}% | Cost: ${total_cost:.3f} | Remain: {remain / 60:.1f}m]")

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}/processed/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a specific model and category.")
    parser.add_argument("model", help="Model name (must be in config)")
    parser.add_argument("--category", choices=["RA_MCQ", "RASA"],
                        default="RA_MCQ", help="Evaluation category")
    parser.add_argument("--runs", type=int, default=30, help="Runs per question")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")

    args = parser.parse_args()

    evaluate_model(args.model, args.category, args.runs, args.temp)