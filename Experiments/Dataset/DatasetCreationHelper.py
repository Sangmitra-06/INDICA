"""
Dataset Consolidator as a Helper for Cultural Dataset Annotation Tool
=====================================================================

This file aggregates multi-level agreement analysis results (intra-regional, 
inter-regional pairwise, and universal) into a unified JSON structure for 
manual annotation. 

Primary Output: A consolidated JSON file and annotation spaces for establishing ground-truth regional responses.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


class DatasetConsolidator:
    def __init__(self):
        self.regions = ["North", "South", "East", "West", "Central"]
        self.region_pairs = [
            "South_North", "South_East", "South_West", "South_Central",
            "North_East", "North_Central", "East_Central",
            "North_West", "East_West",
            "West_Central"
        ]
        # Lookup table for metadata injection (Category -> Subcategory -> Topic). 
        # Populated dynamically from the source questions file.
        self.question_metadata_lookup = {}

    def load_json(self, filepath: str) -> Dict:
        """Safely load JSON data from the specified path, handling file access errors."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}

    def build_question_metadata_lookup(self, questions_file: str) -> Dict[str, Dict]:
        """
        Construct a hierarchical metadata lookup (Category -> Subcategory -> Topic) 
        hashed by question text. This facilitates efficient metadata injection 
        during the consolidation process.
        """
        lookup = {}
        data = self.load_json(questions_file)

        if not data:
            print(f"Warning: Could not load questions file: {questions_file}")
            return lookup

        for category_block in data:
            category = category_block.get("category", "")
            for subcategory_block in category_block.get("subcategories", []):
                subcategory = subcategory_block.get("subcategory", "")
                for topic_block in subcategory_block.get("topics", []):
                    topic = topic_block.get("topic", "")
                    for question in topic_block.get("questions", []):
                        lookup[question] = {
                            "category": category,
                            "subcategory": subcategory,
                            "topic": topic
                        }

        print(f"  Built metadata lookup for {len(lookup)} questions")
        self.question_metadata_lookup = lookup
        return lookup

    def extract_regional_data(self, intra_regional_files: Dict[str, str]) -> Dict[str, Dict]:
        """
        Aggregate intra-regional analysis results.
        
        Parses individual region files to extract:
        - Agreement status (consensus found/not found)
        - Identified cultural concepts
        - Raw participant responses
        - LLM-generated summaries and suggested answers
        
        Returns:
            Dict mapping question_text -> {region -> analysis_data}
        """
        questions_by_text = defaultdict(dict)

        for region, filepath in intra_regional_files.items():
            print(f"  Processing {region}...")
            data = self.load_json(filepath)

            for q in data.get("questions_analyzed", []):
                question_text = q.get("question_text", "")
                if not question_text:
                    continue

                llm_analysis = q.get("llm_analysis", {})

                # Normalize response extraction using the 'answer' field schema.
                raw_responses = []
                for r in q.get("responses", []):
                    if isinstance(r, dict):
                        # Validate response structure before extraction.
                        response_text = r.get("answer", "")
                        if response_text:
                            raw_responses.append(response_text)

                regional_info = {
                    "question_number": q.get("question_number"),
                    "has_agreement": llm_analysis.get("agreement_found", False),
                    "concepts": llm_analysis.get("common_concepts", []),
                    "raw_responses": raw_responses,
                    "summary": llm_analysis.get("summary", ""),
                    "suggested_answer": self._extract_suggested_answers(llm_analysis.get("common_concepts", []))
                }

                questions_by_text[question_text][region] = regional_info

        return questions_by_text

    def _extract_suggested_answers(self, concepts: List[Dict]) -> List[str]:
        """
        Derive a list of candidate answers from extracted cultural concepts.
        Prioritizes the concept name itself, followed by representative quotes if distinct.
        """
        answers = []
        for concept in concepts:
            if not isinstance(concept, dict):
                continue

            # Get the concept name
            concept_name = concept.get("concept", "")
            if concept_name:
                answers.append(concept_name)

            # Optionally include top quotes as examples
            quotes = concept.get("exact_quotes_proof", [])
            if quotes and isinstance(quotes, list):
                # Add first unique quote as example
                unique_quotes = list(set([str(q).lower().strip() for q in quotes if q]))
                if unique_quotes and concept_name and unique_quotes[0] != concept_name.lower():
                    answers.append(unique_quotes[0])

        return list(set([a for a in answers if a]))  # Deduplicate and clean results.

    def extract_pairwise_agreements(self, inter_regional_files: Dict[str, str]) -> Dict[str, Dict]:
        """
        Aggregate inter-regional comparison data.
        Maps questions to their pairwise agreement status (Boolean) and qualitative summaries.
        """
        pairwise_data = defaultdict(dict)

        for pair_name, filepath in inter_regional_files.items():
            print(f"  Processing {pair_name}...")
            try:
                data = self.load_json(filepath)

                for q in data.get("concept_comparisons", []):
                    question_text = q.get("question_text", "")
                    if not question_text:
                        continue

                    comparison_result = q.get("comparison_result", {})
                    agreement = comparison_result.get("inter_regional_agreement")
                    summary = comparison_result.get("agreement_summary", "")

                    # Normalize missing agreement values to False (disagreement/unknown) 
                    # rather than None to ensure consistent downstream boolean logic.
                    if agreement is None:
                        agreement = False
                        if not summary:
                            summary = "No agreement determination made"

                    pairwise_data[question_text][pair_name] = {
                        "agreement": agreement,
                        "summary": summary
                    }

            except Exception as e:
                print(f"    Warning: Could not process {pair_name}: {e}")

        return pairwise_data

    def extract_universal_agreements(self, multi_region_file: str) -> Dict[str, Any]:
        """
        Extract universal agreement metrics from the multi-region analysis file.
        Captures global consensus status and cross-regional matched concepts.
        """
        universal_data = {}

        print(f"  Processing multi-region file...")
        data = self.load_json(multi_region_file)

        for q in data.get("concept_comparisons", []):
            question_text = q.get("question_text", "")
            if not question_text:
                continue

            comparison = q.get("comparison_result", {})

            universal_data[question_text] = {
                "universal_agreement": comparison.get("universal_agreement", False),
                "agreement_type": comparison.get("agreement_type", "none"),
                "matched_concepts": comparison.get("matched_concepts", []),
                "agreement_summary": comparison.get("agreement_summary", "")
            }

        return universal_data

    def consolidate(
            self,
            intra_regional_files: Dict[str, str],
            inter_regional_files: Dict[str, str],
            multi_region_file: str,
            output_file: str = "manual_annotation_helper.json",
            questions_file: str = None
    ):
        """
        Orchestrate the data consolidation pipeline.
        
        1. Build Metadata Lookup
        2. Aggregate Intra-Regional Data
        3. Aggregate Pairwise Inter-Regional Agreements
        4. Aggregate Universal Agreements
        5. Merge all sources into a unified annotation schema.
        """
        # Load question metadata if provided
        if questions_file:
            print("Step 0: Building question metadata lookup...")
            self.build_question_metadata_lookup(questions_file)

        print("Step 1: Extracting regional data...")
        regional_data = self.extract_regional_data(intra_regional_files)
        print(f"  Found {len(regional_data)} unique questions")

        print("\nStep 2: Extracting pairwise agreements...")
        pairwise_data = self.extract_pairwise_agreements(inter_regional_files)

        print("\nStep 3: Extracting universal agreements...")
        universal_data = self.extract_universal_agreements(multi_region_file)

        print("\nStep 4: Consolidating...")
        consolidated = {
            "metadata": {
                "purpose": "Manual annotation helper for establishing regional answers",
                "instructions": "Fill in 'manual_annotation_space' for each question",
                "total_questions": len(regional_data)
            },
            "questions": []
        }

        # Assign unique identifiers to questions for stable referencing.
        for idx, (question_text, regions) in enumerate(sorted(regional_data.items()), 1):
            question_entry = {
                "question_id": f"q_{idx:03d}",
                "question_text": question_text,
                "regional_data": {}
            }

            # Add regional data
            for region in self.regions:
                if region in regions:
                    question_entry["regional_data"][region] = regions[region]
                else:
                    question_entry["regional_data"][region] = {
                        "has_agreement": False,
                        "concepts": [],
                        "raw_responses": [],
                        "suggested_answer": []
                    }

            # Populate pairwise agreements, defaulting to 'False/Not Compared' if data is missing.
            question_entry["pairwise_agreements"] = {}
            for pair in self.region_pairs:
                pair_data = pairwise_data.get(question_text, {}).get(pair)

                if pair_data is not None and isinstance(pair_data, dict):
                    # Data exists and is in correct format
                    agreement = pair_data.get("agreement")
                    summary = pair_data.get("summary", "")

                    # If agreement is None, treat as False (no comparison was done)
                    if agreement is None:
                        agreement = False
                        if not summary:
                            summary = "Question not compared between these regions"

                    question_entry["pairwise_agreements"][pair] = {
                        "agreement": agreement,
                        "summary": summary
                    }
                else:
                    # No data found for this pair (question doesn't exist in comparison file)
                    question_entry["pairwise_agreements"][pair] = {
                        "agreement": False,
                        "summary": "Question not compared between these regions"
                    }

            # Add universal agreement
            universal_info = universal_data.get(question_text, {})
            question_entry["universal_agreement"] = universal_info.get("universal_agreement", False)
            question_entry["matched_concepts_across_regions"] = universal_info.get("matched_concepts", [])
            question_entry["agreement_summary"] = universal_info.get("agreement_summary", "")

            # Initialize manual annotation structure, pre-filling metadata where available.
            metadata = self.question_metadata_lookup.get(question_text, {})
            question_entry["manual_annotation_space"] = {
                "final_answer_north": [],
                "final_answer_south": [],
                "final_answer_east": [],
                "final_answer_west": [],
                "final_answer_central": [],
                "notes": "",
                "question_category": metadata.get("category", ""),
                "question_subcategory": metadata.get("subcategory", ""),
                "question_topic": metadata.get("topic", "")
            }

            consolidated["questions"].append(question_entry)

        # Save
        print(f"\nStep 5: Saving to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        print(f"Consolidation complete! {len(consolidated['questions'])} questions processed.")

        # Print summary statistics
        self._print_summary(consolidated)

    def _print_summary(self, consolidated: Dict):
        """Calculate and display distribution statistics for the consolidated dataset."""
        total = len(consolidated["questions"])

        # Count questions with regional agreement
        regional_agreement_counts = {region: 0 for region in self.regions}
        universal_count = 0
        pairwise_counts = {pair: 0 for pair in self.region_pairs}

        for q in consolidated["questions"]:
            for region in self.regions:
                if q["regional_data"][region]["has_agreement"]:
                    regional_agreement_counts[region] += 1

            if q["universal_agreement"]:
                universal_count += 1

            # Count pairwise agreements
            for pair in self.region_pairs:
                pair_data = q["pairwise_agreements"].get(pair)
                # Handle both old (bool) and new (dict) formats
                if isinstance(pair_data, dict):
                    if pair_data.get("agreement") is True:
                        pairwise_counts[pair] += 1
                elif pair_data is True:
                    pairwise_counts[pair] += 1

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total questions: {total}")

        print(f"\nQuestions with intra-regional agreement:")
        for region, count in regional_agreement_counts.items():
            print(f"  {region:8s}: {count:3d} ({count / total * 100:5.1f}%)")

        print(f"\nQuestions with universal agreement: {universal_count} ({universal_count / total * 100:.1f}%)")

        print(f"\nPairwise agreement counts:")
        for pair, count in sorted(pairwise_counts.items()):
            if count > 0:  # Only show pairs with some agreement
                print(f"  {pair:15s}: {count:3d} ({count / total * 100:5.1f}%)")

        print("=" * 60 + "\n")


# Usage example
if __name__ == "__main__":
    consolidator = DatasetConsolidator()

    # Define your file paths
    intra_regional_files = {
        "North": "north_intra_agreement_llm_assessed.json",
        "South": "south_intra_agreement_llm_assessed.json",
        "East": "east_intra_agreement_llm_assessed.json",
        "West": "west_intra_agreement_llm_assessed.json",
        "Central": "central_intra_agreement_llm_assessed.json"
    }

    inter_regional_files = {
        "South_North": "south_vs_north_concept_analysis_llm.json",
        "South_East": "south_vs_east_concept_analysis_llm.json",
        "South_West": "south_vs_west_concept_analysis_llm.json",
        "South_Central": "south_vs_central_concept_analysis_llm.json",
        "North_East": "north_vs_east_concept_analysis_llm.json",
        "North_Central": "north_vs_central_concept_analysis_llm.json",
        "East_Central": "east_vs_central_concept_analysis_llm.json",
        "North_West": "north_vs_west_concept_analysis_llm.json",
        "East_West": "east_vs_west_concept_analysis_llm.json",
        "West_Central": "west_vs_central_concept_analysis_llm.json",
    }

    multi_region_file = "north_south_east_west_central_analysis_llm.json"

    # Run consolidation
    consolidator.consolidate(
        intra_regional_files=intra_regional_files,
        inter_regional_files=inter_regional_files,
        multi_region_file=multi_region_file,
        output_file="manual_annotation_helper.json"
    )