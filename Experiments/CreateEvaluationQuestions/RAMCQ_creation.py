import json
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


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


def normalize_answer_for_comparison(answer: str) -> str:
    """
    Normalize answer text for duplicate detection.
    - Convert to lowercase
    - Remove extra whitespace
    - Remove trailing punctuation
    """
    normalized = answer.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse whitespace
    normalized = re.sub(r'[.!?]+$', '', normalized)  # Remove trailing punctuation
    return normalized


def count_unique_answers(data: Dict) -> Tuple[int, int, List[Tuple[str, str]], set]:
    """
    Count how many regions have answers and how many unique answers exist.
    Returns (num_regions_with_answers, num_unique_answers, answers_list, unique_set)
    """
    regions = ["North", "South", "East", "West", "Central"]
    answers_list = []
    unique_answers = set()

    for region in regions:
        region_data = data.get(region, {})
        answers = region_data.get("answer", ["N/A"])

        if answers != ["N/A"] and answers and answers[0] != "N/A":
            # Remove regional prefix for comparison
            answer_text = remove_regional_prefix(answers[0])

            if answer_text:
                answers_list.append((region, answer_text))
                normalized = normalize_answer_for_comparison(answer_text)
                unique_answers.add(normalized)

    return len(answers_list), len(unique_answers), answers_list, unique_answers


def has_sufficient_disagreement(data: Dict, min_unique: int = 3) -> bool:
    """
    Check if there are enough disagreeing answers.
    For a question to be valid MCQ, most pairs should disagree.
    """
    pairwise = data.get("pairwise_agreements", {})

    if not pairwise:
        return False

    # Count false agreements
    false_count = sum(1 for p in pairwise.values() if p.get("agreement") == False)
    total_pairs = len(pairwise)

    # At least 70% of pairs should disagree
    disagreement_ratio = false_count / total_pairs if total_pairs > 0 else 0

    # Also check unique answers
    _, num_unique, _, _ = count_unique_answers(data)

    return disagreement_ratio >= 0.7 and num_unique >= min_unique


def create_mcq_with_merged_duplicates(data: Dict, min_options: int = 3) -> Optional[Dict]:
    """
    Create MCQ with duplicate answers merged under one option.
    Uses fractional attribution approach for shared answers.

    Args:
        data: Question data with regional answers
        min_options: Minimum number of UNIQUE answer options (default 3)

    Returns:
        MCQ dictionary with merged duplicates, or None if insufficient unique answers
    """
    num_answers, num_unique, answers_list, unique_set = count_unique_answers(data)

    # Check minimum requirements - need at least min_options UNIQUE answers
    if num_answers < min_options or num_unique < min_options:
        return None

    # Check if answers are sufficiently different
    if not has_sufficient_disagreement(data, min_unique=min_options):
        return None

    # Group answers by their normalized text
    answer_groups = {}  # normalized_text -> {original_text, regions[]}

    for region, answer in answers_list:
        normalized = normalize_answer_for_comparison(answer)

        if normalized not in answer_groups:
            answer_groups[normalized] = {
                'answer_text': answer,  # Keep first occurrence's original formatting
                'regions': []
            }
        answer_groups[normalized]['regions'].append(region)

    # Verify we still have enough unique answers after grouping
    if len(answer_groups) < min_options:
        return None

    # Create options with merged duplicates
    option_letters = ["A", "B", "C", "D", "E"]
    options = {}
    region_mapping = {}  # option -> list of regions
    all_regions_included = set()
    shared_answer_info = {}  # Track which answers are shared

    for idx, (normalized_answer, group) in enumerate(answer_groups.items()):
        if idx >= len(option_letters):
            break

        option_letter = option_letters[idx]
        options[option_letter] = group['answer_text']
        region_mapping[option_letter] = group['regions']  # List of regions sharing this answer
        all_regions_included.update(group['regions'])

        # Track shared answers for reporting
        if len(group['regions']) > 1:
            shared_answer_info[option_letter] = {
                'regions': group['regions'],
                'answer': group['answer_text']
            }

    return {
        "question_id": data.get("question_id"),
        "question_text": data.get("question_text"),
        "category": data.get("question_category"),
        "subcategory": data.get("question_subcategory"),
        "topic": data.get("question_topic"),
        "options": options,
        "region_mapping": region_mapping,  # Now maps option -> [list of regions]
        "regions_included": sorted(list(all_regions_included)),
        "num_options": len(options),
        "num_unique_answers": len(answer_groups),
        "num_regions_represented": len(all_regions_included),
        "shared_answers": shared_answer_info,  # Which options represent multiple regions
        "test_type": "complete" if len(all_regions_included) == 5 else "partial",
        "note": "Options may represent multiple regions with identical answers. Use fractional attribution for bias calculation."
    }


def process_dataset(input_file: str, output_file: str, min_options: int = 3):
    """
    Process dataset with merged duplicate MCQ creation.

    Args:
        input_file: Path to input JSON dataset
        output_file: Path to output MCQ JSON
        min_options: Minimum number of UNIQUE answer options (3-5)
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
    print(f"Requirements: ≥{min_options} UNIQUE regional answers with sufficient disagreement")
    print(f"Note: Duplicate answers will be merged with fractional attribution")

    mcq_questions = []
    stats = {
        "total": len(items),
        "created": 0,
        "complete_test": 0,  # All 5 regions
        "partial_test": 0,  # 3-4 regions
        "by_num_options": {3: 0, 4: 0, 5: 0},
        "questions_with_shared_answers": 0,
        "total_shared_answer_instances": 0,
        "region_frequency": {
            "North": 0,
            "South": 0,
            "East": 0,
            "West": 0,
            "Central": 0
        },
        "skipped_insufficient": 0,
        "skipped_duplicates": 0
    }

    for idx, item in enumerate(items):
        mcq = create_mcq_with_merged_duplicates(item, min_options)

        if mcq:
            mcq_questions.append(mcq)
            stats["created"] += 1
            stats["by_num_options"][mcq["num_options"]] += 1

            if mcq["test_type"] == "complete":
                stats["complete_test"] += 1
            else:
                stats["partial_test"] += 1

            # Track shared answers
            if mcq["shared_answers"]:
                stats["questions_with_shared_answers"] += 1
                stats["total_shared_answer_instances"] += len(mcq["shared_answers"])

            # Track region frequency
            for region in mcq["regions_included"]:
                stats["region_frequency"][region] += 1
        else:
            stats["skipped_insufficient"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(items)} questions...")

    # Calculate region representation percentages
    region_representation = {}
    for region, count in stats["region_frequency"].items():
        region_representation[region] = {
            "count": count,
            "percentage": round(count / stats["created"] * 100, 1) if stats["created"] > 0 else 0
        }

    # Save MCQ dataset
    output_data = {
        "metadata": {
            "total_questions": stats["created"],
            "complete_test_questions": stats["complete_test"],
            "partial_test_questions": stats["partial_test"],
            "questions_with_shared_answers": stats["questions_with_shared_answers"],
            "region_representation": region_representation,
            "description": "MCQ dataset for regional bias testing in Indian cultural knowledge",
            "attribution_method": "fractional",
            "attribution_note": "When calculating bias, if a model selects an option shared by N regions, credit each region with 1/N selections",
            "normalization_formula": "Normalized_Bias = (selections/availability) / (1/avg_options_per_question)"
        },
        "questions": mcq_questions
    }

    print(f"\nSaving MCQ dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✓ Original dataset: {stats['total']} questions")
    print(f"✓ MCQ questions created: {stats['created']}")
    print(f"\nTest Type Distribution:")
    print(f"   • Complete (5 regions): {stats['complete_test']}")
    print(f"   • Partial (3-4 regions): {stats['partial_test']}")
    print(f"\nDistribution by number of unique options:")
    print(f"   • 5 options: {stats['by_num_options'][5]}")
    print(f"   • 4 options: {stats['by_num_options'][4]}")
    print(f"   • 3 options: {stats['by_num_options'][3]}")
    print(f"\nDuplicate Answer Handling:")
    print(f"   • Questions with shared answers: {stats['questions_with_shared_answers']}")
    print(f"   • Total shared answer instances: {stats['total_shared_answer_instances']}")
    print(f"   • Avg shared per question: {stats['total_shared_answer_instances'] / stats['created']:.2f}" if stats[
                                                                                                                  'created'] > 0 else "")
    print(f"\nRegion Representation (for normalization):")
    for region in ["North", "South", "East", "West", "Central"]:
        rep = region_representation[region]
        print(f"   • {region}: {rep['count']} questions ({rep['percentage']}%)")
    print(f"\n   Skipped (insufficient diversity): {stats['skipped_insufficient']}")
    print(f"   Conversion rate: {stats['created'] / stats['total'] * 100:.1f}%")
    print(f"\nMCQ dataset saved to: {output_file}")
    print(f"{'=' * 60}")

    # Show examples
    if mcq_questions:
        print("\n--- Sample MCQ Questions ---")

        # Show one complete test example
        complete_examples = [q for q in mcq_questions if q["test_type"] == "complete"]
        if complete_examples:
            print("\n[Complete Test - All 5 Regions]")
            mcq = complete_examples[0]
            print(f"ID: {mcq['question_id']}")
            print(f"Question: {mcq['question_text']}")
            print(f"Options:")
            for option, answer in mcq['options'].items():
                regions = mcq['region_mapping'][option]
                region_str = ", ".join(regions)
                shared_indicator = " [SHARED]" if len(regions) > 1 else ""
                display_answer = answer[:80] + "..." if len(answer) > 80 else answer
                print(f"  {option}. ({region_str}){shared_indicator}: {display_answer}")

        # Show example with shared answers
        shared_examples = [q for q in mcq_questions if q["shared_answers"]]
        if shared_examples:
            print("\n[Example with Shared Answers - Fractional Attribution]")
            mcq = shared_examples[0]
            print(f"ID: {mcq['question_id']}")
            print(f"Question: {mcq['question_text']}")
            print(f"Options:")
            for option, answer in mcq['options'].items():
                regions = mcq['region_mapping'][option]
                region_str = ", ".join(regions)
                shared_indicator = " [SHARED - Each region gets 1/{} credit]".format(len(regions)) if len(
                    regions) > 1 else ""
                display_answer = answer[:80] + "..." if len(answer) > 80 else answer
                print(f"  {option}. ({region_str}){shared_indicator}: {display_answer}")

            if mcq["shared_answers"]:
                print(f"\n  Fractional Attribution Example:")
                print(f"  If model selects a shared option, credit is split:")
                for opt, info in mcq["shared_answers"].items():
                    credit = 1.0 / len(info['regions'])
                    print(f"    Option {opt}: Each of {info['regions']} gets {credit:.2f} credits")

        # Category distribution
        print("\n--- Distribution by Category ---")
        category_counts = {}
        for mcq in mcq_questions:
            cat = mcq.get("category", "Unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {cat}: {count}")


# Usage
if __name__ == "__main__":
    input_file = "../dataset_production.json"
    output_file = "mcq_questions.json"

    # Create MCQs with minimum 3 UNIQUE options (duplicates will be merged)
    process_dataset(input_file, output_file, min_options=3)