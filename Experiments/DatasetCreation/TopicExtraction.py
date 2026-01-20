"""
Topic Extraction Script
=======================

Extracts culturally grounded topics from OCM (Outline of Cultural Materials) definitions using GPT-4.
Ensures topics reflect socially shared knowledge and normative expectations suitable for
commonsense reasoning benchmarks.
"""
from openai import OpenAI
from openai import AsyncOpenAI

# API Configuration
# Note: Ensure API key is set in environment variables for production use.
api_key = "your-openai-api-key-here"

# Define models and their prices (per 1M tokens)
pricing_per_million = {
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60}
}
# Define models to compare
model = "gpt-4"
# Model selection: GPT-4 chosen for nuanced understanding of anthropology concepts
model = "gpt-4"

client = OpenAI(
  api_key=api_key,
)
# System prompt defines the persona and constraints for cultural topic extraction
system_prompt = """
You are a cultural anthropology expert. Your task is to extract concrete, culturally grounded topics from definitions provided in the Outline of Cultural Materials (OCM), with a focus on commonsense knowledge that reflects everyday norms and expectations within a given society.

These topics will be used to evaluate whether language models possess deep, culturally situated commonsense — the type of knowledge necessary to navigate routine social life in culturally coherent ways.

Goals

Extract topics that:

- Reflect socially shared knowledge (70%+ agreement within a cultural group)
- Are learned through cultural participation, not formal education
- Represent normative expectations, not preferences, frequencies, or trivia
- Are relevant to practical functioning in society — what people *should know* to behave appropriately in common social situations
- Are stable and generalizable across individuals within a cultural group
- Are specific enough to form the basis of a cultural commonsense question (e.g., “What is expected when entering a place of worship?”)

Output Format (Per Topic)

For each topic, return:

1. Topic Label (3–7 words): Concise, clear, culturally grounded
2. Definition: A 1–2 sentence explanation of the commonsense knowledge it reflects within the society
3. Connection to OCM: A sentence showing how the topic derives from specific language or dimensions of the OCM subcategory

Scope and Standards

- Focus on cultural norms, interaction expectations, and implicit social logic that people rely on to function in their communities
- Avoid abstract academic categories, highly individualized behaviors, or edge cases
- Do not include examples or sample questions — your goal is to extract conceptual dimensions, not generate prompts
- Prioritize topics that carry social consequences for incorrect behavior (e.g., shame, respect, offense, admiration)

Cultural Guidance

As you interpret the OCM definition, consider:

- Hierarchical etiquette systems (e.g., age, gender, ritual authority)
- Ritualized or habitual practices around eating, greeting, clothing, interaction
- Moral or symbolic underpinnings of routine social expectations
- Everyday behavioral norms that guide what is appropriate, respectful, or inappropriate
- Local variation, but aim for core practices that are widely shared within the group
"""

# Construct user prompt with specific OCM category context
# TODO: Dynamically inject OCM categories during batch processing
user_prompt="""Please analyze the following OCM entry and extract 8–10 culturally grounded cultural commonsense reasoning topics:

Category: Education

Subcategory: Students

Definition: Composition of the student body; number of general and specialized students in an academic institution; organization and associations (e.g., student unions); leadership; social status of students; sources of financial support (e.g., private funds, scholarships, fellowships, etc.); degree of academic freedom; extracurricular activities; living and dining accommodations (e.g., dormitories, off-campus housing, sororities and fraternities, etc.); student-community relations; student-faculty relationships; group values and behavior characteristics (e.g., political and social activism, etc.); etc.

Focus only on social knowledge that helps people function appropriately in their cultural environment."""
# Prepare message payload
system_message = {"role": "system", "content": system_prompt}
user_message = {"role": "user", "content": user_prompt}

# Execute LLM completion request
# Temperature set to 0.7 to balance creativity in topic formulation with adherence to definition
response = client.chat.completions.create(
    model=model,
    messages=[system_message, user_message],
    temperature=0.7,
    top_p=1.0
)

content = response.choices[0].message.content
usage = response.usage


prompt_tokens = usage.prompt_tokens
completion_tokens = usage.completion_tokens
total_tokens = usage.total_tokens

cost = (prompt_tokens / 1_000_000) * pricing_per_million[model]["prompt"] + (completion_tokens / 1_000_000) * pricing_per_million[model]["completion"]

# Output results and usage metrics
print(f"Response:\n{content}")
print(f"Tokens used - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
print(f"Estimated cost: ${cost:.5f} USD")
