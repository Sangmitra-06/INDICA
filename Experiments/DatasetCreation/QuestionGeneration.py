"""
Cultural Question Generation Script
===================================

Generates culturally grounded commonsense questions based on specific categories and topics using GPT-4.
Utilizes few-shot prompting with region-specific examples to guide the model.
"""
from openai import OpenAI
import os

# API Configuration
# Note: Ensure API key is set in environment variables for production use.
api_key = "your-openai-api-key-here"
model = "gpt-4"
pricing_per_million = {
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60}
}

client = OpenAI(api_key=api_key)

# Define the system role and generation constraints
system_prompt = """
You are a culturally aware commonsense reasoning assistant.

Your task is to generate culturally grounded, realistic questions that reflect everyday social norms, expectations, or interactions within a specific region.

Each question should:
- Be relevant to the provided category, subcategory, and topic definition
- Reflect what someone in that culture is expected to know or understand
- Avoid trivia, preferences, or niche edge cases
- Be open ended and lack specific nouns or indicators for the region
- Must begin with "In your region"
- Take inspiration from the example questions provided for that topic
- Be usable in a cultural commonsense benchmark

Do not include the answer. Focus on what a culturally competent person should ask or understand in social settings.
"""
# Topic definitions (normally populated dynamically during batch processing)
category = "Interpersonal relations"
subcategory = "Visiting and Hospitality"
topic_label = "Etiquette in Reception of Visitors"
topic_definition = "The traditional norms and behaviors associated with receiving and entertaining visitors in a culturally appropriate manner."
region = "India"

# Construct the user prompt with context and few-shot examples
user_prompt = f"""
Category: {category}
Subcategory: {subcategory}
Topic Label: {topic_label}
Topic Definition: {topic_definition}
Region: {region}

Question Examples for this Topic:

1. In your region, what is the first thing you do when you enter someone’s house? Focus on actions and not greetings.
2. In your region, what is a traditional drink, aside from water, that is offered to a guest when they visit your? Be as specific as possible.
3. In your region, what special food items are made when relatives come from out of town? 
4. In your region, how do you traditionally prepare your house for the arrival of guests?
5. In your region, what are the utensils used to serve meals to guests? Are there any changes made from everyday utensils or are different traditional utensils used? 
6. In your region, what are the common customs or expectations when an out-of-town guest arrives, such as from an airport or train station, in terms of how they travel to your home? 

Generate 20-25 culturally grounded commonsense questions that conform to this topic definition and {region}, like the example questions. 

Only output the questions, no bullet points, no commentary.
"""

# Execute LLM completion request
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
    top_p=1.0
)

# Extract content and calculate token cost
content = response.choices[0].message.content
usage = response.usage

prompt_tokens = usage.prompt_tokens
completion_tokens = usage.completion_tokens
total_tokens = usage.total_tokens

cost = (prompt_tokens / 1_000_000) * pricing_per_million[model]["prompt"] + \
       (completion_tokens / 1_000_000) * pricing_per_million[model]["completion"]

# Display results
print("Generated Questions:\n", content)
print(f"Tokens used — Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
print(f"Estimated cost: ${cost:.5f}")
