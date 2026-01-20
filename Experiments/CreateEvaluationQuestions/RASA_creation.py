import json
import re
from typing import Dict, List


def remove_regional_prefix(answer: str) -> str:
    """
    Remove the regional prefix (e.g., "In North India, ") and capitalize first letter.
    """
    # Pattern to match "In [Region] India, " or "In [Region], "
    pattern = r'^In\s+(?:North|South|East|West|Central)\s+India,\s*'
    cleaned = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    # Capitalize first letter if not already
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def create_regional_question(original_question: str, region: str) -> str:
    """
    Create a region-specific version of the question.
    """
    # Check if question already mentions a region (shouldn't, but just in case)
    if "india" in original_question.lower():
        return original_question

    # Make sure the original question starts with lowercase (except first word)
    # First, ensure the first character is lowercase for concatenation
    question_lower = original_question[0].lower() + original_question[1:] if original_question else ""

    # Add regional context at the beginning
    regional_question = f"In {region} India, {question_lower}"

    return regional_question


def process_dataset(input_file: str, output_file: str):
    """
    Convert dataset to region-specific short answer questions.
    """
    print(f"Loading dataset from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if isinstance(dataset, dict):
        if "questions" in dataset:
            items = dataset["questions"]
        elif "data" in dataset:
            items = dataset["data"]
        else:
            items = [dataset]
    elif isinstance(dataset, list):
        items = dataset
    else:
        print("Error: Unexpected dataset format")
        return

    print(f"Processing {len(items)} questions...")

    # Structure: group by region
    regions = ["North", "South", "East", "West", "Central"]
    regional_qa = {region: [] for region in regions}

    stats = {
        "total_questions": len(items),
        "questions_per_region": {region: 0 for region in regions},
        "total_qa_pairs": 0
    }

    for idx, item in enumerate(items):
        original_question = item.get("question_text", "")
        question_id = item.get("question_id", "")
        category = item.get("question_category", "")
        subcategory = item.get("question_subcategory", "")
        topic = item.get("question_topic", "")

        # Process each region
        for region in regions:
            region_data = item.get(region, {})
            answers = region_data.get("answer", ["N/A"])

            # Skip if N/A
            if answers == ["N/A"] or not answers or answers[0] == "N/A":
                continue

            # Create regional question
            regional_question = create_regional_question(original_question, region)

            # Get answer and remove regional prefix
            answer = remove_regional_prefix(answers[0])

            # Add to regional group
            regional_qa[region].append({
                "question_id": f"{question_id}_{region}",
                "original_question_id": question_id,
                "question": regional_question,
                "answer": answer,
                "category": category,
                "subcategory": subcategory,
                "topic": topic
            })

            stats["questions_per_region"][region] += 1
            stats["total_qa_pairs"] += 1

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(items)} questions...")

    # Create output structure
    output_data = {
        "metadata": {
            "description": "Region-specific short answer questions for Indian cultural knowledge",
            "total_questions": stats["total_questions"],
            "total_qa_pairs": stats["total_qa_pairs"],
            "questions_per_region": stats["questions_per_region"],
            "note": "Each question is contextualized to a specific region with the expected answer for that region"
        },
        "North": regional_qa["North"],
        "South": regional_qa["South"],
        "East": regional_qa["East"],
        "West": regional_qa["West"],
        "Central": regional_qa["Central"]
    }

    # Save to file
    print(f"\nSaving short answer dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SHORT ANSWER DATASET CREATED")
    print(f"{'=' * 60}")
    print(f"\nOriginal questions: {stats['total_questions']}")
    print(f"Total Q&A pairs generated: {stats['total_qa_pairs']}")
    print(f"\nQuestions by Region:")
    for region in regions:
        count = stats["questions_per_region"][region]
        percentage = (count / stats["total_qa_pairs"] * 100) if stats["total_qa_pairs"] > 0 else 0
        print(f"   • {region}: {count} ({percentage:.1f}%)")
    print(f"\nDataset saved to: {output_file}")
    print(f"{'=' * 60}")

    # Show examples from each region
    print("\n--- Sample Questions by Region ---")
    for region in regions:
        if regional_qa[region]:
            print(f"\n[{region} India]")
            sample = regional_qa[region][0]
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer'][:100]}..." if len(
                sample['answer']) > 100 else f"Answer: {sample['answer']}")

    # Show category distribution
    print("\n--- Category Distribution ---")
    category_counts = {}
    for region in regions:
        for qa in regional_qa[region]:
            cat = qa.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {cat}: {count}")


# Usage
if __name__ == "__main__":
    input_file = "../dataset_production.json"
    output_file = "short_answer_questions.json"

    process_dataset(input_file, output_file)