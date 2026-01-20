"""
Universal Region Agreement Analyzer
===================================

Analyzes semantic consensus across all defined regions (North, South, East, West, Central).
Identifies "universal" cultural concepts that are shared across all regions, as well as 
finding partial agreement subsets.
"""
import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
from datetime import datetime
import time
import tiktoken
import re


class ConceptBasedInterRegionalAnalyzer:
    def __init__(self):
        # API configuration
        # Note: Consider determining model pricing dynamically or via configuration files.
        self.api_key = "your-openai-api-key-here"
        self.model_name = "gpt-4o"

        # Pricing per million tokens
        self.pricing = {"prompt": 2.50, "completion": 10.00}

        # Token counting
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

        # Initialize OpenAI client
        if self.api_key != "your-api-key-here":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def count_tokens(self, text: str) -> int:
        """Calculate token usage for cost estimation using the model-specific encoder."""
        return len(self.encoder.encode(text))

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute estimated API cost based on current model pricing."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * self.pricing["completion"]
        return prompt_cost + completion_cost

    def normalize_question_text(self, text: str) -> str:
        """
        Normalize text to ensure consistent string matching.
        Removes excess whitespace, lowercases, and strips padding.
        """
        return ' '.join(text.lower().strip().split())

    def load_multiple_regional_analyses(self, results_directory: str, region_names: List[str]) -> Dict[str, Dict]:
        """
        Retrieve and load results for all specified regions.
        Validates file existence before processing.
        """
        regional_data = {}

        for region_name in region_names:
            region_file_prefix = f"{region_name.replace(' ', '_').lower()}_test_results"
            region_path = None

            # Find the actual file
            for file in os.listdir(results_directory):
                if file.startswith(region_file_prefix):
                    region_path = os.path.join(results_directory, file)
                    break

            if not region_path:
                raise FileNotFoundError(f"Could not find results file for {region_name} in {results_directory}")

            print(f"Loading {region_name} analysis from: {region_path}")

            with open(region_path, 'r', encoding='utf-8') as f:
                regional_data[region_name] = json.load(f)

        return regional_data

    def find_multi_region_agreement_questions(self, regional_data: Dict[str, Dict]) -> List[Dict]:
        """
        Identify questions where internal consensus was achieved in EVERY target region.
        Matches questions based on normalized text to handle potential ID mismatches.
        """

        # Collect questions with internal agreement from each region
        # Index questions by normalized text for O(1) matching between datasets.
        region_questions = {}
        for region_name, data in regional_data.items():
            region_questions[region_name] = {}

            for q in data["questions_analyzed"]:
                if "llm_analysis" in q and q["llm_analysis"].get("agreement_found", False):
                    # Normalize question text for matching
                    normalized_text = self.normalize_question_text(q["question_text"])

                    # Store with normalized text as key
                    region_questions[region_name][normalized_text] = {
                        "original_question_text": q["question_text"],
                        "question_number": q["question_number"],
                        "llm_analysis": q["llm_analysis"]
                    }

            print(f"{region_name}: {len(region_questions[region_name])} questions with internal agreement")

        # Find questions present in ALL regions (by question text, not number)
        all_question_texts = set.intersection(*[set(questions.keys()) for questions in region_questions.values()])

        print(f"\nFound {len(all_question_texts)} questions with agreement in ALL {len(regional_data)} regions")

        # Build multi-region question data
        multi_agreement_questions = []
        for normalized_q_text in sorted(all_question_texts):
            # Use the first region's original question text (they should all be the same)
            first_region = list(regional_data.keys())[0]
            original_question_text = region_questions[first_region][normalized_q_text]["original_question_text"]

            # Collect question numbers from each region for reference
            question_numbers_by_region = {
                region_name: region_questions[region_name][normalized_q_text]["question_number"]
                for region_name in regional_data.keys()
            }

            question_data = {
                "question_text": original_question_text,
                "normalized_question_text": normalized_q_text,
                "question_numbers_by_region": question_numbers_by_region,  # Track different Q numbers
                "regional_analyses": {}
            }

            # Add each region's analysis
            for region_name in regional_data.keys():
                question_data["regional_analyses"][region_name] = {
                    "question_number": region_questions[region_name][normalized_q_text]["question_number"],
                    "common_concepts": region_questions[region_name][normalized_q_text]["llm_analysis"][
                        "common_concepts"],
                    "summary": region_questions[region_name][normalized_q_text]["llm_analysis"]["summary"],
                    "agreement_score": region_questions[region_name][normalized_q_text]["llm_analysis"][
                        "agreement_score"]
                }

            multi_agreement_questions.append(question_data)

        # Show summary
        if multi_agreement_questions:
            print(f"\nQuestions with agreement across all regions:")
            for i, q in enumerate(multi_agreement_questions[:10]):
                # Show question numbers from each region
                q_nums_str = ", ".join([f"{region}:Q{num}" for region, num in q["question_numbers_by_region"].items()])
                print(f"[{q_nums_str}]")
                print(f"   {q['question_text'][:100]}...")
                print()

            if len(multi_agreement_questions) > 10:
                print(f"... and {len(multi_agreement_questions) - 10} more questions")

        return multi_agreement_questions

    def create_multi_region_comparison_prompt(self, question_text: str, regional_analyses: Dict[str, Dict]) -> str:
        """
        Construct a comprehensive prompt for simultaneous multi-region comparison.
        Provides the LLM with agreed-upon concepts from all regions to determine universality.
        """

        # Format concepts from all regions
        # Aggregate concept data from all regions for the prompt
        regions_text = ""
        region_names = list(regional_analyses.keys())

        for region_name, analysis in regional_analyses.items():
            regions_text += f"\n{'=' * 60}\n"
            regions_text += f"{region_name.upper()} - AGREED-UPON CONCEPTS\n"
            regions_text += f"{'=' * 60}\n"

            if not analysis["common_concepts"]:
                regions_text += "   No concepts found.\n"
            else:
                for i, concept in enumerate(analysis["common_concepts"], 1):
                    concept_name = concept.get("concept", "Unknown")
                    quotes = concept.get("exact_quotes_proof", [])

                    regions_text += f"\n   Concept {i}: '{concept_name}'\n"
                    regions_text += f"   Supporting Evidence:\n"

                    for j, quote in enumerate(quotes[:4], 1):  # Show up to 4 quotes
                        regions_text += f"      {j}. \"{quote}\"\n"

                    if len(quotes) > 4:
                        regions_text += f"      ... and {len(quotes) - 4} more quotes\n"

        region_list = ", ".join(region_names)
        num_regions = len(region_names)

        prompt = f"""Compare already-identified concepts from {len(regional_analyses)} regions to find inter-regional agreement.

QUESTION: {question_text}

REGIONAL CONCEPTS:
{regions_text}

TASK: Determine if there is agreement across ANY or ALL of these regions: {region_list}
Systematically compare concepts across all {num_regions} regions ({region_list}) to determine:
1. Whether there is UNIVERSAL agreement (all {num_regions} regions share the same concept)
2. Whether there is PARTIAL agreement (some but not all regions share concepts)
3. Whether there is NO agreement (each region has completely different concepts)


INTER-REGIONAL AGREEMENT CRITERIA:
- Universal agreement exists if at least one concept from ALL regions, North, South, East, West, and Central is semantically similar
- Partial agreement exists if at least one concept is semantically similar for some and not ALL 5 regions 
- For specific festivals and traditions, names must be exact matches
- All concepts must answer the same question
- Semantic similarity includes synonyms, variations, and different expressions of the same idea

SEMANTIC MATCHING EXAMPLES:
- "emergency situations" matches "urgent circumstances" (same underlying concept)
- "cleaning house" matches "home tidying" (same activity)
- "touching feet" matches "feet touching" (same gesture)
- "festival sweets" matches "celebratory desserts" (same food category)

NO SEMANTIC MATCHING EXAMPLES: 
- "pongal" and "lohri" are both names of a harvest festival with the first for south and second for north but they are not semantically similar
- "godh bharai" and "valaikappu" are both names of a pregnancy ceremony with the first for south and second for north but they are not semantically similar

NO UNIVERSAL AGREEMENT EXAMPLE
- If North, South, West, and Central regions mention Diwali as the most popular festival but East mentions Durga Puja, that is not counted as universal agreement, instead it is partial agreement


ANALYSIS PROCESS:
STEP 1: CREATE A COMPARISON MATRIX

- List all concepts from all {num_regions} regions
- For each unique concept group, identify which regions mention it
- Example format:
  Concept Group 1: "Fasting/Observing Fast"
    → Present in: North, South, West
  Concept Group 2: "Pongal"
    → Present in: South only
  Concept Group 3: "Lohri"  
    → Present in: North only

STEP 2: APPLY SEMANTIC MATCHING RULES
- Compare each concept from Region A with each concept from Regions B, C, D, E
- Use the matching rules above to determine if concepts are semantically similar
- Remember: Similar category ≠ Semantic match (e.g., both are festivals, but different festivals)

STEP 3: IDENTIFY AGREEMENT PATTERNS
- **Universal Agreement**: At least ONE concept is shared by ALL {num_regions} regions
- **Partial Agreement**: At least ONE concept is shared by SOME regions (2 or more, but not all)
- **No Agreement**: Each region has completely different concepts OR no semantic matches found

STEP 4: DOCUMENT MATCHES
- For each matched concept group, clearly state:
  • The unified concept name (the general term that encompasses all variations)
  • Which regions share this concept
  • What each region calls it (regional variations)
  • Why they are semantically similar (evidence from quotes)

You must return ONLY valid JSON:
{{
    "universal_agreement": true/false,
    "agreement_type": "universal|partial|none",
    "regions_in_agreement": ["region1", "region2", ...],
    "matched_concepts": [
        {{
            "unified_concept_name": "shared concept name",
            "regions_sharing": ["region1", "region2", ...],
            "regional_variations": {{
                "region1": "concept name in region1",
                "region2": "concept name in region2"
            }},
            "semantic_explanation": "How these concepts are semantically similar"
        }}
    ],
    "concept_matrix": [
        {{
            "region1": "concept_name",
            "region2": "concept_name",
            "region3": "concept_name",
            "semantic_match": true/false,
            "explanation": "why they match or don't match"
        }}
    ],
    "agreement_summary": "Brief explanation of inter-regional agreement patterns"
}}"""

        return prompt

    def parse_multi_region_response(self, response_content: str) -> Dict:
        """
        Process the complex multi-region JSON response.
        Handles parsing validation and extracting key metrics like agreement type and confidence.
        """
        if not response_content or response_content.strip() == "":
            return {
                "universal_agreement": False,
                "agreement_type": "none",
                "regions_in_agreement": [],
                "matched_concepts": [],
                "agreement_summary": "Empty response from LLM",
                "parse_error": "Empty response"
            }

        # Clean response
        cleaned = response_content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)

            result = {
                "universal_agreement": bool(parsed.get("universal_agreement", False)),
                "agreement_type": str(parsed.get("agreement_type", "none")),
                "regions_in_agreement": parsed.get("regions_in_agreement", []),
                "matched_concepts": parsed.get("matched_concepts", []),
                "concept_matrix": parsed.get("concept_matrix", []),
                "question_validation": parsed.get("question_validation", {}),
                "step_by_step_analysis": parsed.get("step_by_step_analysis", {}),
                "unmatched_concepts": parsed.get("unmatched_concepts", []),
                "agreement_summary": str(parsed.get("agreement_summary", "No summary provided")),
                "confidence_level": str(parsed.get("confidence_level", "unknown")),
                "ambiguous_cases": parsed.get("ambiguous_cases", []),
                "full_llm_response": parsed,
                "raw_response": response_content
            }

            return result

        except json.JSONDecodeError as e:
            return {
                "universal_agreement": False,
                "agreement_type": "none",
                "regions_in_agreement": [],
                "matched_concepts": [],
                "agreement_summary": f"JSON parse error: {str(e)}",
                "parse_error": str(e),
                "raw_response": response_content
            }

    def compare_multi_region_concepts(self, question_data: Dict) -> Dict:
        """
        Execute comparison logic for a single question across all regions.
        Orchestrates prompt creation, API execution, and results parsing.
        """
        if self.client is None:
            return {
                "universal_agreement": False,
                "agreement_type": "none",
                "regions_in_agreement": [],
                "matched_concepts": [],
                "agreement_summary": "OpenAI client not initialized",
                "api_error": "No API key provided"
            }

        prompt = self.create_multi_region_comparison_prompt(
            question_data["question_text"],
            question_data["regional_analyses"]
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at comparing cultural concepts across multiple regions for semantic similarity. You must respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000  # Increased for multiple regions + validation
            )

            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            response_content = response.choices[0].message.content

            result = self.parse_multi_region_response(response_content)
            result["token_usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost": cost
            }

            return result

        except Exception as e:
            return {
                "universal_agreement": False,
                "agreement_type": "none",
                "regions_in_agreement": [],
                "matched_concepts": [],
                "agreement_summary": f"API error: {str(e)}",
                "api_error": str(e)
            }

    def display_multi_region_comparison(self, question_data: Dict, result: Dict, question_index: int):
        """Print structured multi-region comparison results to the console."""
        print(f"\n" + "=" * 100)
        print(f"MULTI-REGION CONCEPT COMPARISON {question_index}")
        print(f"=" * 100)

        # Show question with region-specific question numbers
        q_nums_str = ", ".join(
            [f"{region}:Q{num}" for region, num in question_data["question_numbers_by_region"].items()])
        print(f"Question Numbers by Region: {q_nums_str}")

        question_text = question_data["question_text"]
        if len(question_text) > 120:
            print(f"QUESTION: {question_text[:120]}...")
        else:
            print(f"QUESTION: {question_text}")

        # Show concepts from each region
        for region_name, analysis in question_data["regional_analyses"].items():
            print(f"\n{region_name.upper()} INTERNAL AGREEMENT (Q{analysis['question_number']}):")
            for i, concept in enumerate(analysis["common_concepts"], 1):
                concept_name = concept.get("concept", "Unknown")
                quotes = concept.get("exact_quotes_proof", [])[:2]
                quotes_str = "', '".join(quotes)
                print(f"   {i}. '{concept_name}' - Evidence: '{quotes_str}'")

        # Show validation results
        validation = result.get("question_validation", {})
        if validation:
            invalid_regions = validation.get("regions_with_invalid_concepts", [])
            if invalid_regions:
                print(f"\nVALIDATION ISSUES:")
                for detail in validation.get("invalid_concept_details", []):
                    print(f"   {detail['region']}: '{detail['concept']}' - {detail['why_invalid']}")

        print(f"\n" + "-" * 70)
        print(f"MULTI-REGION COMPARISON RESULT:")

        universal_agreement = result.get("universal_agreement", False)
        agreement_type = result.get("agreement_type", "none")
        regions_in_agreement = result.get("regions_in_agreement", [])
        cost = result.get("token_usage", {}).get("cost", 0)
        summary = result.get("agreement_summary", "No summary")
        confidence = result.get("confidence_level", "unknown")

        if universal_agreement:
            print(f"   UNIVERSAL AGREEMENT (ALL {len(question_data['regional_analyses'])} REGIONS)")
        elif agreement_type == "partial":
            print(f"   PARTIAL AGREEMENT ({len(regions_in_agreement)} regions)")
            print(f"   Regions in agreement: {', '.join(regions_in_agreement)}")
        else:
            print(f"   NO INTER-REGIONAL AGREEMENT")

        print(f"   Confidence: {confidence}")
        print(f"   Cost: ${cost:.4f}")

        # Show matched concepts
        matched_concepts = result.get("matched_concepts", [])
        if matched_concepts:
            print(f"   MATCHED CONCEPTS:")
            for i, match in enumerate(matched_concepts, 1):
                unified_name = match.get("unified_concept_name", "Unknown")
                regions_sharing = match.get("regions_sharing", [])
                variations = match.get("regional_variations", {})
                explanation = match.get("semantic_explanation", "No explanation")
                rule = match.get("matching_rule_applied", "Unknown")

                print(f"      {i}. Unified Concept: '{unified_name}'")
                print(f"         - Shared by: {', '.join(regions_sharing)}")
                print(f"         - Matching Rule: {rule}")
                for region, variation in variations.items():
                    print(f"         - {region}: '{variation}'")
                print(f"         - Explanation: {explanation}")

        # Show unmatched concepts
        unmatched = result.get("unmatched_concepts", [])
        if unmatched:
            print(f"   UNMATCHED CONCEPTS:")
            for i, unmatch in enumerate(unmatched, 1):
                print(f"      {i}. {unmatch['region']}: '{unmatch['concept_name']}' - {unmatch['why_no_match']}")

        print(f"   SUMMARY: {summary}")

    def analyze_multi_region_concepts(self, results_directory: str, region_names: List[str],
                                      max_questions: Optional[int] = None, save_results: bool = True):
        """
        Main driver function for multi-region analysis.
        Controls the workflow: Loading -> Common Question Identification -> Comparison -> Storage.
        """
        print(f"\n" + "=" * 100)
        print(f"MULTI-REGION CONCEPT-BASED ANALYSIS")
        print(f"Comparing: {', '.join([r.upper() for r in region_names])}")
        print(f"=" * 100)

        try:
            # Load all regional analyses
            regional_data = self.load_multiple_regional_analyses(results_directory, region_names)

            # Find questions with agreement in all regions (BY QUESTION TEXT)
            multi_agreement_questions = self.find_multi_region_agreement_questions(regional_data)

            if not multi_agreement_questions:
                print(f"No questions found where all {len(region_names)} regions have internal agreement")
                return None

            # Limit questions if specified
            if max_questions:
                multi_agreement_questions = multi_agreement_questions[:max_questions]
                print(f"\nAnalyzing first {len(multi_agreement_questions)} questions")

            # Estimate cost
            if multi_agreement_questions:
                sample_prompt = self.create_multi_region_comparison_prompt(
                    multi_agreement_questions[0]["question_text"],
                    multi_agreement_questions[0]["regional_analyses"]
                )
                sample_tokens = self.count_tokens(sample_prompt)
                estimated_cost_per_question = self.calculate_cost(sample_tokens, 600)
                total_estimated_cost = estimated_cost_per_question * len(multi_agreement_questions)

                print(f"Estimated total cost: ${total_estimated_cost:.4f}")

                if len(multi_agreement_questions) > 5:
                    proceed = input(
                        f"\nProceed with analysis? This will cost approximately ${total_estimated_cost:.4f} (y/n): ")
                    if proceed.lower() != 'y':
                        print("Analysis cancelled.")
                        return None

            # Analyze each question
            total_cost = 0
            universal_agreements = 0
            partial_agreements = 0
            no_agreements = 0
            successful_comparisons = 0

            results = {
                "analysis_info": {
                    "timestamp": datetime.now().isoformat(),
                    "regions": region_names,
                    "analysis_type": "multi_region_concept_based",
                    "model": self.model_name,
                    "total_questions_analyzed": len(multi_agreement_questions),
                    "matching_method": "question_text"  # Document that we match by text, not number
                },
                "concept_comparisons": [],
                "summary": {}
            }

            for i, question_data in enumerate(multi_agreement_questions, 1):
                print(f"\nAnalyzing question {i}/{len(multi_agreement_questions)}...")

                comparison_result = self.compare_multi_region_concepts(question_data)

                # Display results
                self.display_multi_region_comparison(question_data, comparison_result, i)

                # Track statistics
                cost = comparison_result.get("token_usage", {}).get("cost", 0)
                total_cost += cost

                if "api_error" not in comparison_result and "parse_error" not in comparison_result:
                    successful_comparisons += 1

                    if comparison_result.get("universal_agreement", False):
                        universal_agreements += 1
                    elif comparison_result.get("agreement_type") == "partial":
                        partial_agreements += 1
                    else:
                        no_agreements += 1

                # Store result
                question_result = {
                    "question_text": question_data["question_text"],
                    "question_numbers_by_region": question_data["question_numbers_by_region"],
                    "regional_concepts": {
                        region: analysis["common_concepts"]
                        for region, analysis in question_data["regional_analyses"].items()
                    },
                    "comparison_result": comparison_result
                }
                results["concept_comparisons"].append(question_result)

                time.sleep(0.5)

            # Final summary
            results["summary"] = {
                "total_questions_analyzed": len(multi_agreement_questions),
                "successful_comparisons": successful_comparisons,
                "universal_agreements": universal_agreements,
                "partial_agreements": partial_agreements,
                "no_agreements": no_agreements,
                "universal_agreement_rate": (
                            universal_agreements / successful_comparisons * 100) if successful_comparisons > 0 else 0,
                "partial_agreement_rate": (
                            partial_agreements / successful_comparisons * 100) if successful_comparisons > 0 else 0,
                "total_cost": total_cost,
                "average_cost_per_question": total_cost / successful_comparisons if successful_comparisons > 0 else 0
            }

            print(f"\n" + "=" * 100)
            print(f"MULTI-REGION ANALYSIS COMPLETE")
            print(f"=" * 100)
            print(f"Total cost: ${total_cost:.4f}")
            print(f"Successful comparisons: {successful_comparisons}")
            print(f"Universal agreements (all {len(region_names)} regions): {universal_agreements}")
            print(f"Partial agreements (some regions): {partial_agreements}")
            print(f"No agreements: {no_agreements}")
            print(
                f"Universal agreement rate: {universal_agreements / successful_comparisons * 100:.1f}%" if successful_comparisons > 0 else "0.0%")

            if save_results:
                self.save_multi_region_results(results, region_names)

            return results

        except Exception as e:
            print(f"Error during multi-region analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_multi_region_results(self, results: Dict, region_names: List[str]):
        """
        Persist results to disk.
        Saves detailed JSON data and a summary CSV for analysis.
        """
        results_dir = "multi_region_concept_results"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        regions_str = "_".join([r.replace(' ', '_').lower() for r in region_names])

        # Save detailed JSON
        json_filename = f"{regions_str}_analysis_{timestamp}.json"
        json_filepath = os.path.join(results_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create summary CSV
        csv_data = []
        for comparison in results["concept_comparisons"]:
            comp_result = comparison["comparison_result"]

            # Count concepts in each region
            concept_counts = {region: len(concepts) for region, concepts in comparison["regional_concepts"].items()}

            # Get matched concepts info
            matched_concepts = comp_result.get("matched_concepts", [])
            regions_in_agreement = comp_result.get("regions_in_agreement", [])

            # Get question numbers by region
            q_nums_by_region = comparison["question_numbers_by_region"]

            row = {
                "Question_Text": comparison["question_text"][:100] + "..." if len(
                    comparison["question_text"]) > 100 else comparison["question_text"],
                "Agreement_Type": comp_result.get("agreement_type", "none"),
                "Universal_Agreement": comp_result.get("universal_agreement", False),
                "Num_Regions_in_Agreement": len(regions_in_agreement),
                "Regions_in_Agreement": "; ".join(regions_in_agreement),
                "Matched_Concepts_Count": len(matched_concepts),
                "Confidence": comp_result.get("confidence_level", "unknown"),
                "Cost": comp_result.get("token_usage", {}).get("cost", 0),
                "Has_Error": "Yes" if ("parse_error" in comp_result or "api_error" in comp_result) else "No"
            }

            # Add question numbers for each region
            for region in region_names:
                row[f"{region}_Q_Number"] = q_nums_by_region.get(region, "N/A")

            # Add concept counts for each region
            for region in region_names:
                row[f"{region}_Concept_Count"] = concept_counts.get(region, 0)

            csv_data.append(row)

        if csv_data:
            csv_filename = f"{regions_str}_summary_{timestamp}.csv"
            csv_filepath = os.path.join(results_dir, csv_filename)

            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filepath, index=False, encoding='utf-8')

        print(f"\nMULTI-REGION RESULTS SAVED:")
        print(f"   Detailed results: {json_filepath}")
        if csv_data:
            print(f"   Summary CSV: {csv_filepath}")
        print(f"   Results directory: {os.path.abspath(results_dir)}")


def main():
    """
    Entry point for the script.
    Configures regions and directories and initiates the multi-region analysis.
    """

    analyzer = ConceptBasedInterRegionalAnalyzer()

    results_directory = "path to your directory"

    # Specify ALL 5 regions
    regions = ["North", "South", "East", "West", "Central"]  # Adjust names as needed

    print("MULTI-REGION CONCEPT-BASED ANALYSIS")
    print(f"Comparing: {', '.join(regions)}")
    print(f"Matching method: By question text (not question number)")

    # Run multi-region analysis
    results = analyzer.analyze_multi_region_concepts(
        results_directory,
        regions,
        max_questions=999,  # Start small to test
        save_results=True
    )


if __name__ == "__main__":
    main()