"""
Evaluation Utilities
====================

Provides helper functions for generating prompts, querying LLM APIs (OpenAI, OpenRouter),
and processing model responses for various evaluation tasks (RASA, RA-MCQ).
"""

from openai import OpenAI
import requests
import random
from typing import Dict, Tuple

# Configuration
# API Configurations
# Note: Use environment variables for sensitive keys in production
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
OPENAI_API_KEY = "your-openai-api-key-here"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_PRICING = {
    "gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.00, "provider": "openai"},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "provider": "openrouter"},
    "anthropic/claude-sonnet-4.5": {"input": 3.00, "output": 15.00, "provider": "openrouter"},
    "x-ai/grok-4-fast": {"input": 0.20, "output": 0.50, "provider": "openrouter"},
    "mistralai/mistral-large-2512": {"input": 0.50, "output": 1.50, "provider": "openrouter"},
    "deepseek/deepseek-v3.2": {"input": 0.26, "output": 0.38, "provider": "openrouter"},
    "qwen/qwen3-vl-235b-a22b-instruct": {"input": 0.20, "output": 1.20, "provider": "openrouter"},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.0, "output": 0.0, "provider": "openrouter"},
}


def create_ramcq_prompt(question: Dict, shuffle_options: bool = True) -> Tuple[str, Dict]:
    """
    Construct a prompt for RA-MCQ (Region-Agnostic Multiple Choice Questions) 
    where the model predicts without regional context.
    Options are optionally shuffled to prevent position bias.
    """
    # Convert options dictionary to list of tuples for processing
    if isinstance(question['options'], dict):
        options = list(question['options'].items())
    else:
        options = list(question['options'].items())

    if shuffle_options:
        random.shuffle(options)

    prompt = f"""You are answering a question about Indian cultural practices. Please select the most appropriate answer from the given options. 

Question: {question['question_text']}

Options:
"""

    new_mapping = {}
    for idx, (orig_letter, text) in enumerate(options):
        new_letter = ['A', 'B', 'C', 'D', 'E'][idx]
        prompt += f"{new_letter}. {text}\n"

        # Normalize region mapping data structure
        regions = question['region_mapping'][orig_letter]
        
        if isinstance(regions, list):
            new_mapping[new_letter] = ", ".join(regions)
        else:
            new_mapping[new_letter] = regions

    prompt += """\nIMPORTANT: Respond with ONLY the letter of your chosen answer (A, B, C, D, or E). Do not provide any explanation or additional text.

Your answer:"""

    return prompt, new_mapping


def create_rasa_prompt(question: Dict) -> str:
    """
    Construct a prompt for RASA (Region-Anchored Short Answer) questions.
    Enforces conciseness to facilitate automated evaluation.
    """
    prompt = f"""You are answering a question about Indian cultural practices. Please provide a concise answer that directly answers the question. 

Question: {question['question']}

IMPORTANT: Provide a direct answer in 1 sentence. Do not use conversational filler or any justifications. Simply answer the question in the most brief way possible. 

Answer:"""
    return prompt


def query_model(prompt: str, model: str, provider: str, temperature: float = 1.0) -> Tuple[str, Dict]:
    """
    Dispatch query to the specified model provider.
    Handles authentication, payload formatting, and response parsing.
    """
    try:
        if provider == "openai":
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            answer = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return answer, usage

        else:  # openrouter
            # Check for free model limits or overrides
            extra_headers = {}
            if "free" in model:
                extra_headers["HTTP-Referer"] = "https://github.com/cultural-commonsense"
                extra_headers["X-Title"] = "Cultural Commonsense Eval"

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    **extra_headers
                },
                json=payload,
                timeout=120  # Increased timeout
            )

            # Handle API errors
            if response.status_code != 200:
                print(f"[Error] API Error {response.status_code}: {response.text}")
                return None, {}

            data = response.json()
            if 'choices' not in data or not data['choices']:
                print(f"[Warning] No choices in response: {data}")
                return None, {}

            return data['choices'][0]['message']['content'].strip(), data.get('usage', {})

    except Exception as e:
        print(f"[Error] Querying {model}: {e}")
        return None, {}


def extract_answer_letter(response: str) -> str:
    """
    Parse the model's response to identify the selected option letter.
    Robustly handles various response formats (single letter, letter with dot, etc.).
    """
    if not response:
        return None

    response = response.strip().upper()
    valid_options = ['A', 'B', 'C', 'D', 'E']

    if response in valid_options:
        return response
    if response and response[0] in valid_options:
        return response[0]

    # Scan for the first valid option letter in the response text
    for char in response:
        if char in valid_options:
            return char
    return None