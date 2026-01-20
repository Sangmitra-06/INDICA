# INDICA Codebase

This folder contains the code for generating, refining, and evaluating on the INDICA dataset. The codebase is organized into four main modules:

## Directory Structure

### 1. DatasetCreation
Scripts for the initial generation of raw cultural data.
*   **`TopicExtraction.py`**: Extracts culturally grounded topics from OCM (Outline of Cultural Materials) definitions using LLMs.
*   **`QuestionGeneration.py`**: Generates culture-specific commonsense questions based on extracted topics and seed questions.

### 2. Dataset
Scripts for analyzing consensus and refining the generated questions.
*   **`IntraRegionAgreement.py`**: Analyzes semantic consensus for questions within a single region (e.g., verifying if North Indian annotators agree).
*   **`InterRegionAgreement.py`**: Compares concepts between two regions to identify shared norms.
*   **`UniversalAgreement.py`**: Analyzes agreement across all regions to identify universally shared cultural concepts.
*   **`datasetAnnotationTool.py`**: Tool for manually reviewing and annotating the dataset.
*   **`DatasetCreationHelper.py`**: Merges intra-regional consensus, pairwise comparisons, and universal agreement analyses into a single annotation-ready JSON file for **`datasetAnnotationTool.py'**.

### 3. CreateEvaluationQuestions
Scripts for transforming approved data into specific evaluation benchmarks.
*   **`RAMCQ_creation.py`**: Generates **Region Agnostic Multiple Choice Questions (RAMCQ)**.
*   **`RASA_creation.py`**: Generates **Region Anchored Short Answer (RASA)** questions for open-ended evaluation.

### 4. Evaluation
Scripts for benchmarking Language Models on the created datasets.
*   **`evaluate_all_models.py`**: The main driver script to evaluate a model on RAMCQ, Regular MCQ, or RASA tasks.
*   **`RASA_LLMAsJudge.py`**: Implements an "LLM-as-a-Judge" framework to evaluate the short-answer responses against gold references.
*   **`RAMCQ_bias_detection.py`**: Analyzes model outputs on RA MCQs to detect regional selection biases.
*   **`ChiSquareTest.py`**: Performs statistical tests (Chi-Square) to validate if model selection patterns are statistically significant.
*   **`results_logger.py`**: Handles structured logging of evaluation results (JSONL, CSV).
*   **`eval_utils.py`**: Shared utilities for prompt generation and API interaction.

## Setup

Ensure you have the necessary dependencies installed. Key libraries include `openai`, `pandas`, `scipy`, and `tiktoken`.

```bash
pip install openai pandas scipy tiktoken
```

## Usage Examples
### Running an Evaluation
To evaluate a model (e.g., GPT-4) on the Adversarial MCQ dataset:
```bash
python evaluate_all_models.py "gpt-4" --category RA_MCQ
```

### Running Bias Analysis
To analyze the regional bias of a model after evaluation:
```bash
python RAMCQ_bias_detection.py
```
