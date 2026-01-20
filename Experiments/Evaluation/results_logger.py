"""
Results Logging Utility
=======================

Handles structured logging of model evaluation results.
Supports multiple output formats (JSONL, CSV, Raw Text) to facilitate
both automated analysis and manual inspection of model performance.
"""

import json
import csv
import os
import time
from typing import Any, Dict, Optional


class ResultsLogger:
    def __init__(self, output_dir: str = "results", category: str = "Unspecified"):
        """
        Initialize the results logger.

        Args:
            output_dir: Root directory for storing evaluation artifacts.
            category: specific evaluation task identifier (e.g., 'RA_MCQ').
        """
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed")
        self.raw_logs_dir = os.path.join(output_dir, "raw_outputs")
        self.category = category

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.raw_logs_dir, exist_ok=True)

    def log_result(self,
                   model_name: str,
                   question_id: str,
                   question_text: str,
                   gold_answer: Any,
                   parsed_answer: Optional[str],
                   raw_response: str,
                   metadata: Dict[str, Any] = None):
        """
        Persist a single evaluation record across all configured formats.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata = metadata or {}

        # Prepare the record
        record = {
            "timestamp": timestamp,
            "model": model_name,
            "category": self.category,
            "question_id": question_id,
            "run_number": metadata.get("run", 1),
            "question_text": question_text,
            "gold_answer": str(gold_answer),
            "parsed_answer": parsed_answer if parsed_answer else "FAILED",
            "extracted_answer": parsed_answer if parsed_answer else "FAILED",  # Duplicate for backward compatibility
            "current_shuffled_mapping": str(metadata.get("mapping", "N/A")),
            "raw_response": raw_response,
            "metadata": metadata
        }

        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        safe_category = self.category.replace(' ', '_').upper()

        # 1. JSONL Logging (Primary Data Source)
        # Stores full metadata and structure for automated analysis
        jsonl_filename = f"{safe_model_name}_{safe_category}_results.jsonl"
        jsonl_path = os.path.join(self.processed_dir, jsonl_filename)

        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. CSV Logging (Human-Readable Summary)
        # Stores a flattened subset of fields for quick inspection
        csv_filename = f"{safe_model_name}_{safe_category}_results.csv"
        csv_path = os.path.join(self.processed_dir, csv_filename)
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "question_id", "category", "run_number",
                "gold_answer", "extracted_answer", "current_shuffled_mapping", "raw_response"
            ])
            if not file_exists:
                writer.writeheader()

            # Write a subset for CSV to keep it readable
            writer.writerow({
                "timestamp": timestamp,
                "question_id": question_id,
                "category": self.category,
                "run_number": record["run_number"],
                "gold_answer": record["gold_answer"],
                "extracted_answer": record["extracted_answer"],
                "current_shuffled_mapping": record["current_shuffled_mapping"],
                "raw_response": raw_response
            })

        # 3. Raw Text Logging (Debug Trace)
        # persist the exact raw response string for debugging parsers
        raw_filename = f"{safe_model_name}_{safe_category}_raw.txt"
        raw_log_path = os.path.join(self.raw_logs_dir, raw_filename)
        with open(raw_log_path, 'a', encoding='utf-8') as f:
            f.write(f"--- {timestamp} | {question_id} ---\n")
            f.write(f"PROMPT_RESPONSE:\n{raw_response}\n")
            f.write("-" * 40 + "\n")

