"""
Cross-Regional Concept Agreement Analyzer
=========================================

Identify and analyze semantic agreement between cultural concepts across different regions.
This module processes pre-computed intra-regional analysis results to find cross-regional
conceptual similarities using LLM-based semantic matching.
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
        # API Key configuration
        # Note: Consider loading from environment variables for security in production deployments.
        self.api_key = "your-openai-api-key-here"
        self.model_name = "gpt-4o"

        # Pricing per million tokens
        self.pricing = {"prompt": 2.50, "completion": 10.00}

        # Token counting
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

        # Initialize OpenAI client
        if self.api_key != "your-openai-api-key-here":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def count_tokens(self, text: str) -> int:
        """Calculate token usage for cost estimation using the model-specific encoder."""
        return len(self.encoder.encode(text))

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost"""
        prompt_cost = (prompt_tokens / 1_000_000) * self.pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * self.pricing["completion"]
        return prompt_cost + completion_cost

    def normalize_question_text(self, text: str) -> str:
        """
        Normalize text to ensure consistent string matching.
        Removes excess whitespace, lowercases, and strips padding.
        """
        return ' '.join(text.lower().strip().split())

    def load_regional_analyses(self, results_directory: str, region1_name: str, region2_name: str) -> Tuple[Dict, Dict]:
        """
        Retrieve and load the JSON result files for specified regions.
        Handles file discovery within the results directory.
        """
        region1_file = f"{region1_name.replace(' ', '_').lower()}_test_results.json"
        region2_file = f"{region2_name.replace(' ', '_').lower()}_test_results.json"

        # Try different possible file patterns
        possible_patterns = [
            f"{region1_name.replace(' ', '_').lower()}_test_results_*.json",
            f"{region1_name.replace(' ', '_').lower()}_*.json"
        ]

        region1_path = None
        region2_path = None

        # Find the actual files
        for file in os.listdir(results_directory):
            if file.startswith(f"{region1_name.replace(' ', '_').lower()}_test_results"):
                region1_path = os.path.join(results_directory, file)
            elif file.startswith(f"{region2_name.replace(' ', '_').lower()}_test_results"):
                region2_path = os.path.join(results_directory, file)

        if not region1_path:
            raise FileNotFoundError(f"Could not find results file for {region1_name} in {results_directory}")
        if not region2_path:
            raise FileNotFoundError(f"Could not find results file for {region2_name} in {results_directory}")

        print(f"Loading {region1_name} analysis from: {region1_path}")
        print(f"Loading {region2_name} analysis from: {region2_path}")

        with open(region1_path, 'r', encoding='utf-8') as f:
            region1_data = json.load(f)

        with open(region2_path, 'r', encoding='utf-8') as f:
            region2_data = json.load(f)

        return region1_data, region2_data

    def find_dual_agreement_questions(self, region1_data: Dict, region2_data: Dict) -> List[Dict]:
        """
        Identify common questions where both regions have independently established internal consensus.
        Matches questions based on normalized text to handle potential ID mismatches.
        """
        print("\n" + "=" * 80)
        print("FINDING QUESTIONS WITH DUAL INTERNAL AGREEMENT")
        print("=" * 80)

        # Index questions by normalized text for O(1) matching between datasets.
        region1_questions = {}
        for q in region1_data["questions_analyzed"]:
            if "llm_analysis" in q and q["llm_analysis"].get("agreement_found", False):
                normalized_text = self.normalize_question_text(q["question_text"])
                region1_questions[normalized_text] = {
                    "original_question_text": q["question_text"],
                    "question_number": q["question_number"],
                    "llm_analysis": q["llm_analysis"]
                }

        region2_questions = {}
        for q in region2_data["questions_analyzed"]:
            if "llm_analysis" in q and q["llm_analysis"].get("agreement_found", False):
                normalized_text = self.normalize_question_text(q["question_text"])
                region2_questions[normalized_text] = {
                    "original_question_text": q["question_text"],
                    "question_number": q["question_number"],
                    "llm_analysis": q["llm_analysis"]
                }

        print(f"{region1_data['analysis_info']['region']}: {len(region1_questions)} questions with internal agreement")
        print(f"{region2_data['analysis_info']['region']}: {len(region2_questions)} questions with internal agreement")

        # Perform intersection on normalized question text to identify comparative candidates.
        dual_agreement_questions = []
        common_question_texts = set(region1_questions.keys()).intersection(set(region2_questions.keys()))

        for normalized_q_text in sorted(common_question_texts):
            region1_q = region1_questions[normalized_q_text]
            region2_q = region2_questions[normalized_q_text]

            dual_agreement_questions.append({
                "question_text": region1_q["original_question_text"],
                "normalized_question_text": normalized_q_text,
                "region1_question_number": region1_q["question_number"],
                "region2_question_number": region2_q["question_number"],
                "region1_analysis": {
                    "region": region1_data['analysis_info']['region'],
                    "question_number": region1_q["question_number"],
                    "common_concepts": region1_q["llm_analysis"]["common_concepts"],
                    "summary": region1_q["llm_analysis"]["summary"],
                    "agreement_score": region1_q["llm_analysis"]["agreement_score"]
                },
                "region2_analysis": {
                    "region": region2_data['analysis_info']['region'],
                    "question_number": region2_q["question_number"],
                    "common_concepts": region2_q["llm_analysis"]["common_concepts"],
                    "summary": region2_q["llm_analysis"]["summary"],
                    "agreement_score": region2_q["llm_analysis"]["agreement_score"]
                }
            })

        print(f"\nFound {len(dual_agreement_questions)} questions with agreement in BOTH regions")

        # Show summary of questions found
        if dual_agreement_questions:
            print(f"\nQuestions with dual internal agreement:")
            for i, q in enumerate(dual_agreement_questions[:10]):  # Show first 10
                print(
                    f"[{region1_data['analysis_info']['region']}:Q{q['region1_question_number']} vs {region2_data['analysis_info']['region']}:Q{q['region2_question_number']}]")
                print(f"   {q['question_text'][:80]}...")
                print()

            if len(dual_agreement_questions) > 10:
                print(f"... and {len(dual_agreement_questions) - 10} more questions")

        return dual_agreement_questions

    def create_concept_comparison_prompt(self, question_text: str, region1_name: str, region1_concepts: List[Dict],
                                         region2_name: str, region2_concepts: List[Dict]) -> str:
        """
        Construct a structured prompt for the LLM to evaluate semantic similarity between concepts.
        Provides the model with extracted concepts and evidence from both regions for a specific question.
        """

        # Format region 1 concepts
        region1_concept_text = ""
        for i, concept in enumerate(region1_concepts, 1):
            concept_name = concept.get("concept", "Unknown")
            quotes = concept.get("exact_quotes_proof", [])
            quotes_str = "', '".join(quotes[:3])  # Show first 3 quotes
            region1_concept_text += f"   {i}. Concept: '{concept_name}'\n      Evidence: '{quotes_str}'\n"

        # Format region 2 concepts
        region2_concept_text = ""
        for i, concept in enumerate(region2_concepts, 1):
            concept_name = concept.get("concept", "Unknown")
            quotes = concept.get("exact_quotes_proof", [])
            quotes_str = "', '".join(quotes[:3])  # Show first 3 quotes
            region2_concept_text += f"   {i}. Concept: '{concept_name}'\n      Evidence: '{quotes_str}'\n"

        prompt = f"""Compare already-identified concepts from two regions to find inter-regional agreement.

QUESTION: {question_text}

{region1_name.upper()} AGREED-UPON CONCEPTS (from within-region analysis):
{region1_concept_text}

{region2_name.upper()} AGREED-UPON CONCEPTS (from within-region analysis):
{region2_concept_text}

TASK: Determine if any concept from {region1_name} matches any concept from {region2_name}.

INTER-REGIONAL AGREEMENT CRITERIA:
- Agreement exists if ANY concept from {region1_name} is semantically similar to ANY concept from {region2_name}
- For specific festivals and traditions, the festival or tradition names have to be an exact match for agreement
- Both concepts must answer the same question
- Semantic similarity includes synonyms, variations, and different ways of expressing the same idea

SEMANTIC MATCHING EXAMPLES:
- "emergency situations" matches "urgent circumstances" (same underlying concept)
- "cleaning house" matches "home tidying" (same activity)
- "touching feet" matches "feet touching" (same gesture)
- "festival sweets" matches "celebratory desserts" (same food category)
- "August" matches "august" matches "month of August" (same month)

NO SEMANTIC MATCHING EXAMPLES: 
- "pongal" and "lohri" are both names of a harvest festival with the first for south and second for north but they are not semantically similar
- "godh bharai" and "valaikappu" are both names of a pregnancy ceremony with the first for south and second for north but they are not semantically similar

ANALYSIS PROCESS:
1. Compare each {region1_name} concept with each {region2_name} concept
2. Look for semantic similarity in the concept names and evidence quotes
3. If any pair matches, inter-regional agreement exists
4. If no concepts match, no inter-regional agreement

You must return ONLY valid JSON:
{{
    "concept_comparisons": [
        {{
            "region1_concept": "concept name from {region1_name}",
            "region2_concept": "concept name from {region2_name}",
            "semantic_match": true,
            "matching_explanation": "Why these concepts represent the same underlying idea"
        }}
    ],
    "inter_regional_agreement": true,
    "matched_concepts": [
        {{
            "unified_concept_name": "shared concept name",
            "region1_concept": "original concept from {region1_name}",
            "region2_concept": "original concept from {region2_name}",
            "semantic_explanation": "How these concepts are semantically similar"
        }}
    ],
    "agreement_summary": "Brief explanation of whether and why inter-regional agreement was found"
}}

REMEMBER: You are comparing already-verified concepts that showed internal agreement within each region. Look for semantic similarity between these established concepts."""

        return prompt

    def parse_concept_comparison_response(self, response_content: str) -> Dict:
        """
        Process the LLM's JSON response.
        Handles cleaning of potential markdown formatting and validates structured output.
        """
        if not response_content or response_content.strip() == "":
            return {
                "inter_regional_agreement": False,
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
                "inter_regional_agreement": bool(parsed.get("inter_regional_agreement", False)),
                "matched_concepts": parsed.get("matched_concepts", []),
                "concept_comparisons": parsed.get("concept_comparisons", []),
                "agreement_summary": str(parsed.get("agreement_summary", "No summary provided")),
                "full_llm_response": parsed,
                "raw_response": response_content
            }

            return result

        except json.JSONDecodeError as e:
            return {
                "inter_regional_agreement": False,
                "matched_concepts": [],
                "agreement_summary": f"JSON parse error: {str(e)}",
                "parse_error": str(e),
                "raw_response": response_content
            }

    def compare_concepts(self, question_data: Dict) -> Dict:
        """
        Execute the concept comparison logic for a single question.
        Orchestrates prompt generation, API call, and response parsing.
        """
        if self.client is None:
            return {
                "inter_regional_agreement": False,
                "matched_concepts": [],
                "agreement_summary": "OpenAI client not initialized",
                "api_error": "No API key provided"
            }

        question_text = question_data["question_text"]
        region1_name = question_data["region1_analysis"]["region"]
        region1_concepts = question_data["region1_analysis"]["common_concepts"]
        region2_name = question_data["region2_analysis"]["region"]
        region2_concepts = question_data["region2_analysis"]["common_concepts"]

        prompt = self.create_concept_comparison_prompt(
            question_text, region1_name, region1_concepts, region2_name, region2_concepts
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at comparing cultural concepts for semantic similarity. You must respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            usage = response.usage
            cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            response_content = response.choices[0].message.content

            result = self.parse_concept_comparison_response(response_content)
            result["token_usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost": cost
            }

            return result

        except Exception as e:
            return {
                "inter_regional_agreement": False,
                "matched_concepts": [],
                "agreement_summary": f"API error: {str(e)}",
                "api_error": str(e)
            }

    def display_question_comparison(self, question_data: Dict, result: Dict, question_index: int):
        """Print formatted results of the comparison to the console."""
        print(f"\n" + "=" * 100)
        print(f"CONCEPT COMPARISON {question_index}")
        print(f"=" * 100)

        # Show region-specific question numbers
        print(
            f"Question Numbers: {question_data['region1_analysis']['region']}:Q{question_data['region1_question_number']} vs {question_data['region2_analysis']['region']}:Q{question_data['region2_question_number']}")

        question_text = question_data["question_text"]
        if len(question_text) > 120:
            print(f"QUESTION: {question_text[:120]}...")
        else:
            print(f"QUESTION: {question_text}")

        region1_name = question_data["region1_analysis"]["region"]
        region2_name = question_data["region2_analysis"]["region"]

        print(f"COMPARING: {region1_name} vs {region2_name}")

        # Show concepts from each region
        print(f"\n{region1_name.upper()} INTERNAL AGREEMENT (Q{question_data['region1_question_number']}):")
        for i, concept in enumerate(question_data["region1_analysis"]["common_concepts"], 1):
            concept_name = concept.get("concept", "Unknown")
            quotes = concept.get("exact_quotes_proof", [])[:2]  # Show first 2
            quotes_str = "', '".join(quotes)
            print(f"   {i}. '{concept_name}' - Evidence: '{quotes_str}'")

        print(f"\n{region2_name.upper()} INTERNAL AGREEMENT (Q{question_data['region2_question_number']}):")
        for i, concept in enumerate(question_data["region2_analysis"]["common_concepts"], 1):
            concept_name = concept.get("concept", "Unknown")
            quotes = concept.get("exact_quotes_proof", [])[:2]  # Show first 2
            quotes_str = "', '".join(quotes)
            print(f"   {i}. '{concept_name}' - Evidence: '{quotes_str}'")

        print(f"\n" + "-" * 70)
        print(f"CONCEPT COMPARISON RESULT:")

        agreement = result.get("inter_regional_agreement", False)
        cost = result.get("token_usage", {}).get("cost", 0)
        summary = result.get("agreement_summary", "No summary")

        if agreement:
            print(f"   INTER-REGIONAL AGREEMENT FOUND")
        else:
            print(f"   NO INTER-REGIONAL AGREEMENT")

        print(f"   Cost: ${cost:.4f}")

        if "parse_error" in result:
            print(f"   Parse Error: {result['parse_error']}")
        elif "api_error" in result:
            print(f"   API Error: {result['api_error']}")

        # Show matched concepts if found
        matched_concepts = result.get("matched_concepts", [])
        if matched_concepts:
            print(f"   MATCHED CONCEPTS:")
            for i, match in enumerate(matched_concepts, 1):
                unified_name = match.get("unified_concept_name", "Unknown")
                region1_concept = match.get("region1_concept", "Unknown")
                region2_concept = match.get("region2_concept", "Unknown")
                explanation = match.get("semantic_explanation", "No explanation")

                print(f"      {i}. Unified Concept: '{unified_name}'")
                print(f"         - {region1_name}: '{region1_concept}'")
                print(f"         - {region2_name}: '{region2_concept}'")
                print(f"         - Why similar: {explanation}")

        print(f"   SUMMARY: {summary}")

    def analyze_inter_regional_concepts(self, results_directory: str, region1_name: str, region2_name: str,
                                        max_questions: Optional[int] = None, save_results: bool = True):
        """
        Main driver function for inter-regional analysis.
        Controls the workflow: Data Loading -> Comparison Logic -> Results Aggregation -> Storage.
        """
        print(f"\n" + "=" * 100)
        print(f"CONCEPT-BASED INTER-REGIONAL ANALYSIS")
        print(f"Comparing: {region1_name.upper()} vs {region2_name.upper()}")
        print(f"Matching method: By question text (not question number)")
        print(f"=" * 100)

        try:
            # Load regional analyses
            region1_data, region2_data = self.load_regional_analyses(results_directory, region1_name, region2_name)

            # Find questions with dual agreement (BY QUESTION TEXT)
            dual_agreement_questions = self.find_dual_agreement_questions(region1_data, region2_data)

            if not dual_agreement_questions:
                print("No questions found where both regions have internal agreement")
                return None

            # Limit questions if specified
            if max_questions:
                dual_agreement_questions = dual_agreement_questions[:max_questions]
                print(f"\nAnalyzing first {len(dual_agreement_questions)} questions")

            # Estimate cost
            if dual_agreement_questions:
                sample_prompt = self.create_concept_comparison_prompt(
                    dual_agreement_questions[0]["question_text"],
                    dual_agreement_questions[0]["region1_analysis"]["region"],
                    dual_agreement_questions[0]["region1_analysis"]["common_concepts"],
                    dual_agreement_questions[0]["region2_analysis"]["region"],
                    dual_agreement_questions[0]["region2_analysis"]["common_concepts"]
                )
                sample_tokens = self.count_tokens(sample_prompt)
                estimated_cost_per_question = self.calculate_cost(sample_tokens, 300)
                total_estimated_cost = estimated_cost_per_question * len(dual_agreement_questions)

                print(f"Estimated total cost: ${total_estimated_cost:.4f}")

                if len(dual_agreement_questions) > 5:
                    proceed = input(
                        f"\nProceed with analysis? This will cost approximately ${total_estimated_cost:.4f} (y/n): ")
                    if proceed.lower() != 'y':
                        print("Analysis cancelled.")
                        return None

            # Analyze each question
            total_cost = 0
            inter_regional_agreements = 0
            successful_comparisons = 0

            results = {
                "analysis_info": {
                    "timestamp": datetime.now().isoformat(),
                    "region1": region1_name,
                    "region2": region2_name,
                    "analysis_type": "concept_based_inter_regional_mongo",
                    "model": self.model_name,
                    "matching_method": "question_text",
                    "total_questions_with_dual_agreement": len(dual_agreement_questions),
                    "questions_analyzed": len(dual_agreement_questions)
                },
                "concept_comparisons": [],
                "summary": {}
            }

            for i, question_data in enumerate(dual_agreement_questions, 1):
                print(f"\nAnalyzing question {i}/{len(dual_agreement_questions)}...")

                comparison_result = self.compare_concepts(question_data)

                # Display results
                self.display_question_comparison(question_data, comparison_result, i)

                # Track statistics
                cost = comparison_result.get("token_usage", {}).get("cost", 0)
                total_cost += cost

                if "api_error" not in comparison_result and "parse_error" not in comparison_result:
                    successful_comparisons += 1
                    if comparison_result.get("inter_regional_agreement", False):
                        inter_regional_agreements += 1
                        print(f"   COUNTED AS INTER-REGIONAL AGREEMENT")
                    else:
                        print(f"   COUNTED AS NO INTER-REGIONAL AGREEMENT")

                # Store result
                question_result = {
                    "question_text": question_data["question_text"],
                    "region1_question_number": question_data["region1_question_number"],
                    "region2_question_number": question_data["region2_question_number"],
                    "region1_concepts": question_data["region1_analysis"]["common_concepts"],
                    "region2_concepts": question_data["region2_analysis"]["common_concepts"],
                    "comparison_result": comparison_result
                }
                results["concept_comparisons"].append(question_result)

                time.sleep(0.5)  # Be nice to API

            # Final summary
            results["summary"] = {
                "total_dual_agreement_questions": len(dual_agreement_questions),
                "successful_comparisons": successful_comparisons,
                "inter_regional_agreements": inter_regional_agreements,
                "inter_regional_agreement_rate": (
                        inter_regional_agreements / successful_comparisons * 100) if successful_comparisons > 0 else 0,
                "total_cost": total_cost,
                "average_cost_per_question": total_cost / successful_comparisons if successful_comparisons > 0 else 0
            }

            print(f"\n" + "=" * 100)
            print(f"CONCEPT-BASED INTER-REGIONAL ANALYSIS COMPLETE")
            print(f"=" * 100)
            print(f"Total cost: ${total_cost:.4f}")
            print(f"Questions with both regions having internal agreement: {len(dual_agreement_questions)}")
            print(f"Successful concept comparisons: {successful_comparisons}")
            print(f"Inter-regional agreements found: {inter_regional_agreements}/{successful_comparisons}")
            print(
                f"Inter-regional agreement rate: {inter_regional_agreements / successful_comparisons * 100:.1f}%" if successful_comparisons > 0 else "Inter-regional agreement rate: 0.0%")

            if save_results:
                self.save_concept_comparison_results(results, region1_name, region2_name)

            return results

        except Exception as e:
            print(f"Error during concept-based analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_concept_comparison_results(self, results: Dict, region1_name: str, region2_name: str):
        """
        Persist results to disk.
        Saves detailed JSON data and a summary CSV for analysis.
        """
        results_dir = "concept_based_inter_regional_results_mongo"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        region1_clean = region1_name.replace(' ', '_').lower()
        region2_clean = region2_name.replace(' ', '_').lower()

        # Save detailed JSON
        json_filename = f"{region1_clean}_vs_{region2_clean}_concept_analysis_{timestamp}.json"
        json_filepath = os.path.join(results_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create summary CSV
        csv_data = []
        for comparison in results["concept_comparisons"]:
            comp_result = comparison["comparison_result"]

            # Count concepts in each region
            region1_concept_count = len(comparison["region1_concepts"])
            region2_concept_count = len(comparison["region2_concepts"])

            # Get matched concepts info
            matched_concepts = comp_result.get("matched_concepts", [])
            matched_concept_names = [mc.get("unified_concept_name", "Unknown") for mc in matched_concepts]

            row = {
                "Question_Text": comparison["question_text"][:100] + "..." if len(
                    comparison["question_text"]) > 100 else comparison["question_text"],
                "Region1_Q_Number": comparison["region1_question_number"],
                "Region2_Q_Number": comparison["region2_question_number"],
                "Inter_Regional_Agreement": comp_result.get("inter_regional_agreement", False),
                "Region1_Concept_Count": region1_concept_count,
                "Region2_Concept_Count": region2_concept_count,
                "Matched_Concepts_Count": len(matched_concepts),
                "Matched_Concept_Names": "; ".join(matched_concept_names) if matched_concept_names else "None",
                "Cost": comp_result.get("token_usage", {}).get("cost", 0),
                "Has_Error": "Yes" if ("parse_error" in comp_result or "api_error" in comp_result) else "No",
                "Agreement_Summary": comp_result.get("agreement_summary", "")[:150] + "..." if len(
                    comp_result.get("agreement_summary", "")) > 150 else comp_result.get("agreement_summary", "")
            }
            csv_data.append(row)

        if csv_data:
            csv_filename = f"{region1_clean}_vs_{region2_clean}_concept_summary_{timestamp}.csv"
            csv_filepath = os.path.join(results_dir, csv_filename)

            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filepath, index=False, encoding='utf-8')

        print(f"\nCONCEPT-BASED RESULTS SAVED:")
        print(f"   Detailed results: {json_filepath}")
        if csv_data:
            print(f"   Summary CSV: {csv_filepath}")
        print(f"   Results directory: {os.path.abspath(results_dir)}")


def main():
    """
    Entry point for the script. 
    Configures regions and directories and initiates the analysis processing.
    """

    # Initialize analyzer
    analyzer = ConceptBasedInterRegionalAnalyzer()

    # Directory containing your within-region analysis results
    results_directory = "path to your folder"  # UPDATE THIS PATH

    # Specify the two regions to compare
    region1 = "South"
    region2 = "North"

    print("CONCEPT-BASED INTER-REGIONAL ANALYSIS")
    print(f"This approach uses existing within-region analyses to compare concepts")
    print(f"Comparing: {region1} vs {region2}")
    print(f"Matching method: By question text (not question number)")

    # Run concept-based inter-regional analysis
    results = analyzer.analyze_inter_regional_concepts(
        results_directory,
        region1,
        region2,
        max_questions=620,  # Start with 10 questions for testing
        save_results=True
    )


if __name__ == "__main__":
    main()