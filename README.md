# 🇮🇳 INDICA

This is the repository for [Common to _Whom_? Regional Cultural Commonsense and LLM Bias in India](https://arxiv.org/abs/2510.20543). 

**📦 Dataset:** Available on [Hugging Face](https://huggingface.co/datasets/Sangmitra-06/CENTERBENCH)

Authors: Sangmitra Madhusudan, Trush Shashank More, Steph Buongiorno, Renata Dividino, Jad Kabbara and Ali Emami

## 📄 Paper abstract

Existing cultural commonsense benchmarks treat nations as _monolithic_, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce **INDICA**, the first benchmark designed to test LLMs' ability to address this question, focusing on India—a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly _regional_, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%–20.9% accuracy on region-specific questions, and they exhibit  geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement.

## 📁 Repository Structure
```
.
├── Dataset/
│   ├── dataset_slice.json
│   └── full_dataset.json
└── Experiments/
    ├── DatasetCreation/
    │   ├── TopicExtraction.py
    │   └── QuestionGeneration.py
    ├── Dataset/
    │   ├── IntraRegionAgreement.py
    │   ├── InterRegionAgreement.py
    │   ├── UniversalAgreement.py
    │   ├── datasetAnnotationTool.py
    │   └── DatasetCreationHelper.py
    ├── CreateEvaluationQuestions/
    │   ├── RAMCQ_creation.py
    │   └── RASA_creation.py
    └── Evaluation/
        ├── evaluate_all_models.py
        ├── RASA_LLMAsJudge.py
        ├── RAMCQ_bias_detection.py
        ├── ChiSquareTest.py
        ├── results_logger.py
        └── eval_utils.py
```

## 🗃️ Dataset

The `Dataset/` folder contains the INDICA dataset files:

- **`dataset.json`**: Complete dataset with all 515 culturally grounded questions

### Dataset Structure

The dataset is provided in JSON format. 

#### Hierarchy & Metadata
- **`question_id`**: Unique identifier (e.g., `q_001`)
- **`question_text`**: The cultural commonsense question
- **Category Hierarchy**:
  - `question_category`: High-level domain (e.g., *Interpersonal Relations*, *Education*)
  - `question_subcategory`: Subcategory within the domain
  - `question_topic`: The granular topic being addressed

#### Regional Responses
For each of the five regions (**North**, **South**, **East**, **West**, **Central**):
- **`answer`**: A list containing the culturally validated answer(s) for that region. If a region does not have a consensus for that specific question, it is marked as "N/A"
- **`pairwise_agreements`**: Boolean flags showing agreement between every pair of regions (e.g., `South_North`, `East_West`)
- **`universal_agreement`**: Boolean flag indicating if **all** 5 regions agree on the answer

### Example Entry
```json
{
    "question_id": "q_012",
    "question_text": "Are tinted windows allowed on vehicles, and what types of vehicles most commonly have them?",
    "question_category": "Traffic and transport behavior",
    "question_subcategory": "Streets and traffic",
    "question_topic": "Understanding Local Traffic Regulations",
    "North": {
      "answer": [
        "In North India, tinted windows are not allowed on vehicles."
      ]
    },
    "South": {
      "answer": [
        "In South India, tinted windows are not allowed on vehicles."
      ]
    },
    "East": {
      "answer": [
        "In East India, tinted windows are not allowed on vehicles."
      ]
    },
    "West": {
      "answer": [
        "In West India, tinted windows are not allowed on vehicles."
      ]
    },
    "Central": {
      "answer": [
        "In Central India, tinted windows are not allowed on vehicles."
      ]
    },
    "pairwise_agreements": {
      "South_North": {
        "agreement": true
      },
      "South_East": {
        "agreement": true
      },
      "South_West": {
        "agreement": true
      },
      "South_Central": {
        "agreement": true
      },
      "North_East": {
        "agreement": true
      },
      "North_Central": {
        "agreement": true
      },
      "East_Central": {
        "agreement": true
      },
      "North_West": {
        "agreement": true
      },
      "East_West": {
        "agreement": true
      },
      "West_Central": {
        "agreement": true
      }
    },
    "universal_agreement": true
}
```
## 🔧 Experiments

The `Experiments/` folder contains scripts for dataset generation, analysis, and model evaluation, organized into four main modules:

### Dataset Creation (`Experiments/DatasetCreation/`)

Scripts for the initial generation of topics and questions:

- **`TopicExtraction.py`**: Extracts culturally grounded topics from OCM (Outline of Cultural Materials) definitions using LLMs
- **`QuestionGeneration.py`**: Generates culture-specific commonsense questions based on extracted topics and seed questions

### Dataset Refinement (`Experiments/Dataset/`)

Scripts for analyzing consensus and annotation:

- **`IntraRegionAgreement.py`**: Analyzes semantic consensus for questions within a single region (e.g., verifying if North Indian annotators agree)
- **`InterRegionAgreement.py`**: Compares concepts between two regions to identify shared norms
- **`UniversalAgreement.py`**: Analyzes agreement across all regions to identify universally shared cultural concepts
- **`datasetAnnotationTool.py`**: Tool for manually reviewing and annotating the dataset
- **`DatasetCreationHelper.py`**: Merges intra-regional consensus, pairwise comparisons, and universal agreement analyses into a single annotation-ready JSON file

### Evaluation Question Creation (`Experiments/CreateEvaluationQuestions/`)

Scripts for transforming the dataset into specific evaluation benchmarks:

- **`RAMCQ_creation.py`**: Generates **Region Agnostic Multiple Choice Questions (RAMCQ)** for bias detection
- **`RASA_creation.py`**: Generates **Region Anchored Short Answer (RASA)** questions for open-ended evaluation

### Model Evaluation (`Experiments/Evaluation/`)

Scripts for benchmarking Language Models on the dataset:

- **`evaluate_all_models.py`**: Main driver script to evaluate a model on RAMCQ, Regular MCQ, or RASA tasks
- **`RASA_LLMAsJudge.py`**: Implements an "LLM-as-a-Judge" framework to evaluate short-answer responses against gold references
- **`RAMCQ_bias_detection.py`**: Analyzes model outputs on RA MCQs to detect regional selection biases
- **`ChiSquareTest.py`**: Performs Chi-Square to validate if model selection patterns are statistically significant
- **`results_logger.py`**: Handles structured logging of evaluation results (JSONL, CSV)
- **`eval_utils.py`**: Shared utilities for prompt generation and API interaction

## 🖥️ Usage

### 1. Clone the Repository

Clone this repository to access the complete INDICA dataset:
```bash
git clone https://github.com/Sangmitra-06/INDICA.git
cd INDICA
```

### 2. Install Dependencies

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
# Add other API keys as needed
```

### 4. Load the Dataset
```python
import json

# Load the full dataset
with open('Dataset/dataset.json', 'r') as f:
    indica_data = json.load(f)

# Access questions by domain
traffic&transport_questions = [q for q in indica_data if q['question_category'] == 'Traffic & Transport Behavior']
```

### 5. Run Evaluation

Use our evaluation scripts to test models on the dataset:
```bash
# Evaluate on Region Agnostic Multiple Choice Questions (RAMCQ)
python Experiments/Evaluation/evaluate_all_models.py "gpt-5" --category RA_MCQ

# Evaluate on Region Anchored Short Answer (RASA)
python Experiments/Evaluation/evaluate_all_models.py "gpt-5" --category RASA

# Run bias analysis after evaluation
python Experiments/Evaluation/RAMCQ_bias_detection.py
```
## ✏️ Citation

Please use the following bibtex citation if this paper was a part of your work, thank you!
```bibtex
@misc{madhusudan2025commonwhomregionalcultural,
      title={Common to Whom? Regional Cultural Commonsense and LLM Bias in India}, 
      author={Sangmitra Madhusudan and Trush Shashank More and Steph Buongiorno and Renata Dividino and Jad Kabbara and Ali Emami},
      year={2025},
      eprint={2510.20543},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.20543}, 
}
```
