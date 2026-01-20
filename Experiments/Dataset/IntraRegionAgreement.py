"""
Intra-Regional Agreement Analyzer
=================================

Analyzes consensus within specific cultural regions using LLM-based semantic matching.
This module processes qualitative survey responses to identify shared cultural concepts
and determines if a sufficient agreement threshold is met among participants.
"""
import json
import os
import pandas as pd
from typing import Dict, List, Any
from openai import OpenAI
from datetime import datetime
import time
import tiktoken
import re


class SimpleLLMAgreementAnalyzer:
    def __init__(self):
        # REPLACE WITH YOUR ACTUAL API KEY
        # API configuration
        self.api_key = "your-openai-api-key-here"
        self.model_name = "gpt-4o"

        # Pricing per million tokens
        self.pricing = {"prompt": 2.50, "completion": 10.00}

        # Token counting
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

        # Initialize OpenAI client (new format)
        if self.api_key != "your-openai-api-key-here":
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def count_tokens(self, text: str) -> int:
        """Calculate token usage using the model-specific encoder to estimate costs."""
        return len(self.encoder.encode(text))

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute estimated API cost based on current model pricing per million tokens."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * self.pricing["completion"]
        return prompt_cost + completion_cost

    def create_prompt(self, question_text: str, responses: List[Dict], region: str) -> str:
        """
        Construct a structured analysis prompt.
        
        Dynamically adjusts the agreement threshold based on the number of available responses
        to ensure statistical relevance (e.g., higher threshold for larger sample sizes).
        """
        response_count = len(responses)
        responses_text = "\n".join([f"Response {i + 1}: {resp['answer']}" for i, resp in enumerate(responses)])

        threshold = 4 if response_count >= 5 else max(2, response_count - 1)

        prompt = f"""Analyze agreement among {response_count} responses from {region} about regional practices.

        QUESTION: {question_text}

        NUMBERED RESPONSES:
        {responses_text}

        CORE INSTRUCTION: Only look for concepts that directly answer the question asked. This is imperative above all else. 

        ANALYSIS RULES:
        1. Look for the SAME underlying concept across responses (semantic similarity counts)
        2. Spelling variations, spacing differences, and synonyms count as the SAME concept
        3. You must quote exact text but recognize when different words mean the same thing
        4. {threshold}+ different responses must mention the same underlying concept

        CONCEPT IDENTIFICATION EXAMPLES:
        Question: "What foods are eaten during festivals?"
        - Response mentions "sweets" → Answers the question 
        - Response mentions "celebration" → Doesn't answer what food 

        Question: "What nonverbal actions are disrespectful to elders?"  
        - Response mentions "pointing" → Answers the question (specific action) 
        - Response mentions "being rude" → Doesn't answer what action 
        -
        Question: "In your region, what is the customary gesture for offering food to a deity, and how is it performed?"
        - Response: "We offer food to deity's by putting it on a clean plate" → Concept extracted should be "putting it on a clean plate" since that is what answers the question
        - Response: "We offer food to deity's by putting it on a clean plate" → Concept extracted should NOT be "food is offered", that is not the answer to the question 

        SEMANTIC MATCHING EXAMPLES:
        - "Raksha Bandhan" = "Rakshabandhan" = "rakhi" (same festival)
        - "clean house" = "cleaning home" = "tidy up house" (same activity)
        - "Diwali" = "Deepawali" (same festival)
        - "new clothes" = "fresh clothing" = "new garments" (same concept)

        STEP-BY-STEP ANALYSIS:
        1. Analyse if the response is even answering the question before moving on to getting the concepts
        2. Extract concepts from each response with exact quotes that answer the question 
        3. Group semantically similar concepts together (consider spelling, synonyms, variants)
        4. Count how many different responses mention each concept group
        5. Agreement exists if any concept group appears in {threshold}+ responses

        VERIFICATION FORMAT:
        For each concept group, show the evidence and reasoning.

        You must return ONLY valid JSON:
        {{
            "step_by_step_extraction": {{
                "response_1_concepts": ["concept from response 1"],
                "response_2_concepts": ["concept from response 2"],
                "response_3_concepts": ["concept from response 3"],
                "response_4_concepts": ["concept from response 4"]
            }},
            "semantic_grouping": {{
                "concept_group_name": {{
                    "responses_and_quotes": {{
                        "1": "exact quote from response 1",
                        "2": "exact quote from response 2", 
                        "3": "exact quote from response 3"
                    }},
                    "semantic_explanation": "Why these quotes represent the same concept",
                    "count": 3
                }}
            }},
            "agreement_found": true,
            "threshold_met": "X out of {response_count} responses mention the same underlying concept",
            "common_concepts": [
                {{
                    "concept": "unified concept name",
                    "responses_mentioning": [1, 2, 3],
                    "exact_quotes_proof": ["quote 1", "quote 2", "quote 3"],
                    "semantic_note": "Explanation of any spelling/synonym variations"
                }}
            ],
            "summary": "Brief explanation recognizing semantic similarity while showing evidence"
        }}

        REMEMBER: Only count responses that give the same answer to what the question asks. Use exact quotes as evidence BUT recognize when different spellings/words refer to the same underlying concept. Be reasonable about semantic matching while maintaining evidence requirements.
        NOTE: Please make sure the concepts that are extracted answer the question. """

        return prompt

    def clean_llm_response(self, answer: str) -> str:
        """
        Sanitize LLM output to extract valid JSON.
        Removes Markdown code blocks and locates the JSON object boundaries.
        """
        if not answer or answer.strip() == "":
            return ""

        # Remove any markdown formatting
        cleaned = answer.strip()

        # Remove ```json and ``` markers
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Find JSON object in the response
        cleaned = cleaned.strip()

        # Try to find JSON object boundaries
        start_idx = cleaned.find('{')
        if start_idx == -1:
            return ""

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == '{':
                brace_count += 1
            elif cleaned[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return cleaned[start_idx:]  # Return what we have

        return cleaned[start_idx:end_idx + 1]

    def parse_llm_response(self, response_content: str) -> Dict:
        """
        Parse and validate the JSON response from the LLM.
        Includes error handling for incomplete or malformed data to trigger retry logic.
        """
        if not response_content or response_content.strip() == "":
            return {
                "agreement_found": False,
                "agreement_score": 1,
                "threshold_met": "Could not analyze - empty response",
                "common_concepts": [],
                "summary": "LLM returned empty response",
                "parse_error": "Empty response",
                "full_llm_response": None,
                "raw_response_content": response_content,
                "needs_retry": True
            }

        # Clean the response
        cleaned_response = self.clean_llm_response(response_content)

        if not cleaned_response:
            return {
                "agreement_found": False,
                "agreement_score": 1,
                "threshold_met": "Could not analyze - no JSON found",
                "common_concepts": [],
                "summary": "Could not extract JSON from response",
                "parse_error": "No JSON found",
                "full_llm_response": None,
                "raw_response_content": response_content[:200],
                "needs_retry": True
            }

        # Try to parse JSON
        try:
            parsed = json.loads(cleaned_response)

            # Validate required fields
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")

            required_fields = ["agreement_found", "common_concepts", "summary"]
            missing_fields = [field for field in required_fields if field not in parsed]

            if missing_fields:
                return {
                    "agreement_found": False,
                    "agreement_score": 1,
                    "threshold_met": "Incomplete response",
                    "common_concepts": [],
                    "summary": f"Incomplete response - missing fields: {missing_fields}",
                    "parse_error": f"Missing required fields: {missing_fields}",
                    "needs_retry": True,
                    "partial_response": parsed,
                    "raw_response_content": response_content
                }

            # Check if common_concepts is empty when agreement_found is True
            if parsed.get("agreement_found") and not parsed.get("common_concepts"):
                return {
                    "agreement_found": parsed.get("agreement_found"),
                    "agreement_score": parsed.get("agreement_score", 1),
                    "threshold_met": "Agreement found but concepts list empty",
                    "common_concepts": [],
                    "summary": "Agreement found but concepts list is empty - likely truncated response",
                    "parse_error": "Empty common_concepts despite agreement_found=true",
                    "needs_retry": True,
                    "partial_response": parsed,
                    "raw_response_content": response_content
                }

            # Extract basic fields for backward compatibility with display functions
            basic_fields = {
                "agreement_found": bool(parsed.get("agreement_found", False)),
                "agreement_score": int(parsed.get("agreement_score", 1)),
                "threshold_met": str(parsed.get("threshold_met", "Not specified")),
                "common_concepts": parsed.get("common_concepts", []),
                "summary": str(parsed.get("summary", "No summary provided"))
            }

            # Store the FULL LLM response with all new fields
            result = {
                **basic_fields,
                "full_llm_response": parsed,
                "step_by_step_extraction": parsed.get("step_by_step_extraction", {}),
                "semantic_grouping": parsed.get("semantic_grouping", {}),
                "raw_response_content": response_content,
                "needs_retry": False  # Success!
            }

            return result

        except json.JSONDecodeError as e:
            error_msg = str(e)

            # Check if it's a truncation error
            is_truncated = "Unterminated string" in error_msg or "Expecting" in error_msg

            fallback_result = {
                "agreement_found": self._extract_agreement_from_text(response_content),
                "agreement_score": self._extract_score_from_text(response_content),
                "threshold_met": "Could not parse JSON response",
                "common_concepts": [],
                "summary": f"JSON parse error: {error_msg}",
                "parse_error": error_msg,
                "full_llm_response": None,
                "step_by_step_extraction": {},
                "semantic_grouping": {},
                "cleaned_response": cleaned_response,
                "raw_response_content": response_content,
                "needs_retry": is_truncated  # Retry if truncated
            }
            return fallback_result

    def _extract_agreement_from_text(self, text: str) -> bool:
        """Attempt to extract boolean agreement status from unstructured text using keyword matching."""
        text_lower = text.lower()
        if "agreement_found" in text_lower:
            if "true" in text_lower:
                return True
        if "agreement" in text_lower and ("yes" in text_lower or "found" in text_lower):
            return True
        return False

    def _extract_score_from_text(self, text: str) -> int:
        """Attempt to extract numerical agreement score (1-5) from unstructured text."""
        score_pattern = r"score[\":\s]*(\d)"
        match = re.search(score_pattern, text.lower())
        if match:
            try:
                score = int(match.group(1))
                return score if 1 <= score <= 5 else 1
            except:
                pass
        return 1

    def analyze_question_with_retry(self, question: Dict, region: str, max_retries: int = 5) -> Dict:
        """
        Execute analysis for a single question with progressive retry logic.
        Increases token limits on subsequent attempts to handle truncation issues.
        """

        # Define token limits for each attempt: 800 → 1200 → 1500 → 2000 → 2500 → 3000
        token_limits = [800, 1200, 1500, 2000, 2500, 3000]

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"   [Retry] Attempt {attempt}/{max_retries}")
                time.sleep(1)

            # Get max_tokens for this attempt
            max_tokens = token_limits[min(attempt, len(token_limits) - 1)]

            prompt = self.create_prompt(question['question_text'], question['responses'], region)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert at analyzing cultural agreement. You must respond with valid, complete JSON only, no additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens
                )

                usage = response.usage
                cost = self.calculate_cost(usage.prompt_tokens, usage.completion_tokens)
                response_content = response.choices[0].message.content

                # Parse and validate response
                llm_result = self.parse_llm_response(response_content)
                llm_result["token_usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "cost": cost,
                    "attempt": attempt + 1,
                    "max_tokens_used": max_tokens
                }

                # Check if retry is needed
                if not llm_result.get("needs_retry", False):
                    if attempt > 0:
                        print(f"   [Success] Retry successful on attempt {attempt + 1} (using {max_tokens} max tokens)")
                    return llm_result, cost

                # If this was the last attempt, return with error info
                if attempt == max_retries:
                    print(f"   [Failure] All {max_retries + 1} retry attempts failed")
                    llm_result["retry_exhausted"] = True
                    return llm_result, cost
                else:
                    print(
                        f"   [Warning] Attempt {attempt + 1} failed, will retry with {token_limits[min(attempt + 1, len(token_limits) - 1)]} tokens")

            except Exception as e:
                print(f"   [Error] API error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries:
                    error_result = {
                        "agreement_found": False,
                        "agreement_score": 1,
                        "threshold_met": "API error",
                        "common_concepts": [],
                        "summary": f"API error after {max_retries + 1} attempts: {str(e)}",
                        "api_error": str(e),
                        "retry_exhausted": True
                    }
                    return error_result, 0
                time.sleep(2)

        # Should never reach here
        return {
            "agreement_found": False,
            "common_concepts": [],
            "summary": "Unknown error in retry logic",
            "retry_exhausted": True
        }, 0

    def display_question_analysis(self, question: Dict, region: str, question_index: int):
        """Print question context and responses to console for verification."""
        print(f"\n" + "=" * 80)
        print(f"QUESTION {question_index}: Q{question['question_number']}")
        print(f"=" * 80)

        question_text = question['question_text']
        if len(question_text) > 150:
            print(f"📝 QUESTION: {question_text[:150]}...")
        else:
            print(f"QUESTION: {question_text}")
        print(f"REGION: {region}")
        print(f"RESPONSES: {len(question['responses'])}")

        print(f"\nALL RESPONSES:")
        for i, resp in enumerate(question['responses'], 1):
            answer = resp['answer']
            if len(answer) > 200:
                print(f"   {i}. {answer[:200]}...")
            else:
                print(f"   {i}. {answer}")

        if question.get('cultural_commonsense', {}).get('available', False):
            cs = question['cultural_commonsense']
            print(f"\nCULTURAL COMMONSENSE: {cs['consensus_level']} ({cs['yes_percentage']:.1f}% said YES)")

        print(f"\n" + "-" * 50)
        print(f"LLM ANALYSIS:")

    def display_llm_result(self, llm_result: Dict, cost: float):
        """Print structured analysis results including agreement status, score, and cost."""

        agreement = llm_result.get('agreement_found', False)
        score = llm_result.get('agreement_score', 1)
        summary = llm_result.get('summary', 'No summary provided')

        if agreement:
            print(f"   [AGREEMENT] Score: {score}/5")
        else:
            print(f"   [NO AGREEMENT] Score: {score}/5")

        print(f"   Cost: ${cost:.4f}")

        if 'parse_error' in llm_result:
            print(f"   [Parse Warning] {llm_result['parse_error']}")
            if llm_result.get("retry_exhausted"):
                print(f"   [Failure] RETRY EXHAUSTED - May have incomplete data")

        threshold_info = llm_result.get('threshold_met', 'Not specified')
        print(f"   Threshold: {threshold_info}")

        common_concepts = llm_result.get('common_concepts', [])
        if common_concepts and len(common_concepts) > 0:
            print(f"   COMMON CONCEPTS FOUND:")
            for i, concept in enumerate(common_concepts, 1):
                if isinstance(concept, dict):
                    concept_name = concept.get('concept', 'Unknown concept')
                    responses_mentioning = concept.get('responses_mentioning', [])
                    exact_quotes = concept.get('exact_quotes_proof', [])

                    print(f"      {i}. '{concept_name}'")
                    print(f"         - Mentioned in responses: {responses_mentioning}")
                    if exact_quotes:
                        quotes_str = "', '".join(exact_quotes[:3])
                        print(f"         - Example quotes: '{quotes_str}'")
                else:
                    print(f"      {i}. {concept}")
        else:
            print(f"   NO COMMON CONCEPTS IDENTIFIED")

        print(f"   LLM EXPLANATION: {summary}")

    def test_cost_for_region(self, json_directory: str, region_name: str, num_questions: int = 10):
        """Perform a dry-run to estimate API costs for a subset of questions."""
        print(f"=" * 80)
        print(f"COST ESTIMATION: {region_name.upper()} - First {num_questions} Questions")
        print(f"=" * 80)

        region_file = f"{region_name.replace(' ', '_').lower()}_llm_analysis.json"
        filepath = os.path.join(json_directory, region_file)

        if not os.path.exists(filepath):
            print(f"[Error] File not found: {filepath}")
            available_files = [f for f in os.listdir(json_directory) if f.endswith('_llm_analysis.json')]
            print(f"Available files: {available_files}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data['questions_for_llm_analysis'][:num_questions]

        print(f"Region: {region_name}")
        print(f"Questions to test: {len(questions)}")
        print(f"Model: {self.model_name}")

        total_prompt_tokens = 0
        total_completion_tokens_est = 0

        print(f"\nToken breakdown (estimated):")
        for i, question in enumerate(questions, 1):
            prompt = self.create_prompt(
                question['question_text'],
                question['responses'],
                region_name
            )

            prompt_tokens = self.count_tokens(prompt)
            completion_tokens_est = 400  # Increased estimate
            cost = self.calculate_cost(prompt_tokens, completion_tokens_est)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens_est += completion_tokens_est

            print(
                f"Q{question['question_number']:3d}: {prompt_tokens:4d} prompt + {completion_tokens_est:3d} completion = ${cost:.4f}")

        total_cost = self.calculate_cost(total_prompt_tokens, total_completion_tokens_est)

        print(f"\n" + "-" * 60)
        print(f"ESTIMATED TOTAL FOR {num_questions} QUESTIONS:")
        print(f"   Prompt tokens: {total_prompt_tokens:,}")
        print(f"   Completion tokens: {total_completion_tokens_est:,}")
        print(f"   Total cost: ${total_cost:.4f}")

        all_questions_count = len(data['questions_for_llm_analysis'])
        if all_questions_count > num_questions:
            full_cost = total_cost * (all_questions_count / num_questions)
            print(f"\nEXTRAPOLATED FOR ALL {all_questions_count} QUESTIONS:")
            print(f"   Estimated total cost for {region_name}: ${full_cost:.4f}")

        return total_cost

    def run_actual_test(self, json_directory: str, region_name: str, num_questions: int = 3, save_results: bool = True):
        """
        Execute API calls for the specified questions.
        Manages cost tracking, retries, and result aggregation.
        """
        if self.client is None:
            print("[Error] Please set your actual OpenAI API key first!")
            return

        print(f"\n" + "=" * 80)
        print(f"ACTUAL API TEST: {region_name.upper()} - {num_questions} Questions")
        print(f"=" * 80)
        print("[Warning] This will make real API calls and cost money!")

        region_file = f"{region_name.replace(' ', '_').lower()}_llm_analysis.json"
        filepath = os.path.join(json_directory, region_file)

        if not os.path.exists(filepath):
            print(f"[Error] File not found: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        available_questions = len(data['questions_for_llm_analysis'])
        if num_questions > available_questions:
            print(
                f"[Warning] Requested {num_questions} questions, but only {available_questions} available. Using all {available_questions}.")
            num_questions = available_questions

        questions = data['questions_for_llm_analysis'][:num_questions]

        total_cost = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        agreement_count = 0
        successful_analyses = 0
        retry_successes = 0
        retry_failures = 0

        test_results = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "region": region_name,
                "model": self.model_name,
                "questions_requested": num_questions,
                "questions_analyzed": 0,
                "test_type": "sample_test" if num_questions < available_questions else "full_analysis",
                "retry_enabled": True,
                "max_retries": 5
            },
            "questions_analyzed": [],
            "summary": {}
        }

        for i, question in enumerate(questions, 1):
            self.display_question_analysis(question, region_name, i)

            print(f"   Calling {self.model_name}...")

            # Use retry-enabled analysis
            llm_result, cost = self.analyze_question_with_retry(question, region_name,
                                                                max_retries=5)
            # Display result
            self.display_llm_result(llm_result, cost)

            # Track statistics
            total_cost += cost

            token_usage = llm_result.get("token_usage", {})
            total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            total_completion_tokens += token_usage.get("completion_tokens", 0)

            if "api_error" not in llm_result:
                successful_analyses += 1

                if llm_result.get('agreement_found', False):
                    agreement_count += 1
                    print(f"   [Result] COUNTED AS AGREEMENT")
                else:
                    print(f"   [Result] COUNTED AS NO AGREEMENT")

                if llm_result.get("token_usage", {}).get("attempt", 1) > 1:
                    if not llm_result.get("retry_exhausted"):
                        retry_successes += 1
                    else:
                        retry_failures += 1

            # Store result
            question_result = {
                "question_number": question['question_number'],
                "question_text": question['question_text'],
                "responses": question['responses'],
                "cultural_commonsense": question.get('cultural_commonsense', {}),
                "llm_analysis": llm_result,
                "token_usage": token_usage,
                "raw_llm_response": llm_result.get("raw_response_content", ""),
                "prompt_sent_to_llm": self.create_prompt(question['question_text'], question['responses'], region_name)
            }
            test_results["questions_analyzed"].append(question_result)

            time.sleep(0.5)

        # Update analysis info
        test_results["analysis_info"]["questions_analyzed"] = successful_analyses

        # Create final summary
        test_results["summary"] = {
            "total_questions_requested": num_questions,
            "successfully_analyzed": successful_analyses,
            "questions_with_agreement": agreement_count,
            "agreement_percentage": (agreement_count / successful_analyses * 100) if successful_analyses > 0 else 0,
            "retry_successes": retry_successes,
            "retry_failures": retry_failures,
            "total_cost": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "average_cost_per_question": total_cost / successful_analyses if successful_analyses > 0 else 0
        }

        # Final summary display
        print(f"\n" + "=" * 80)
        print(f"📊 FINAL RESULTS FOR {successful_analyses} SUCCESSFULLY ANALYZED QUESTIONS")
        print(f"=" * 80)
        print(f"💰 COSTS:")
        print(f"   Total prompt tokens: {total_prompt_tokens:,}")
        print(f"   Total completion tokens: {total_completion_tokens:,}")
        print(f"   Total cost: ${total_cost:.4f}")
        print(
            f"   Average cost per question: ${total_cost / successful_analyses:.4f}" if successful_analyses > 0 else "   Average cost per question: $0.0000")

        print(f"\n" + "=" * 80)
        print(f"FINAL RESULTS FOR {successful_analyses} SUCCESSFULLY ANALYZED QUESTIONS")
        print(f"=" * 80)
        print(f"COSTS:")
        print(f"   Total prompt tokens: {total_prompt_tokens:,}")
        print(
            f"   Agreement rate: {agreement_count / successful_analyses * 100:.1f}%" if successful_analyses > 0 else "   Agreement rate: 0.0%")

        print(f"\nRETRY STATISTICS:")
        print(f"   Successful retries: {retry_successes}")
        print(f"   Failed retries: {retry_failures}")

        # Extrapolate
        all_questions_count = len(data['questions_for_llm_analysis'])
        if all_questions_count > num_questions:
            full_cost = total_cost * (all_questions_count / successful_analyses) if successful_analyses > 0 else 0
            full_agreement = agreement_count * (
                        all_questions_count / successful_analyses) if successful_analyses > 0 else 0
            print(f"\nEXTRAPOLATED FOR ALL {all_questions_count} QUESTIONS:")
            print(f"   Estimated total cost for {region_name}: ${full_cost:.4f}")
            print(
                f"   Estimated agreements: {full_agreement:.0f}/{all_questions_count} ({full_agreement / all_questions_count * 100:.1f}%)" if all_questions_count > 0 else "   Estimated agreements: 0/0 (0.0%)")

        if save_results:
            self.save_test_results(test_results, region_name)

        return total_cost, test_results

    def save_test_results(self, results: Dict, region_name: str):
        """Persist analysis results to JSON and CSV formats."""
        results_dir = "test"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        region_clean = region_name.replace(' ', '_').lower()

        json_filename = f"{region_clean}_test_results_{timestamp}.json"
        json_filepath = os.path.join(results_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        csv_data = []
        for q in results["questions_analyzed"]:
            if "llm_analysis" in q:
                llm_analysis = q["llm_analysis"]
                row = {
                    "Question_Number": q["question_number"],
                    "Agreement_Found": llm_analysis.get("agreement_found", False),
                    "Agreement_Score": llm_analysis.get("agreement_score", 1),
                    "Common_Concepts_Count": len(llm_analysis.get("common_concepts", [])),
                    "Cost": q["token_usage"].get("cost", 0),
                    "Prompt_Tokens": q["token_usage"].get("prompt_tokens", 0),
                    "Completion_Tokens": q["token_usage"].get("completion_tokens", 0),
                    "Retry_Attempt": q["token_usage"].get("attempt", 1),
                    "Parse_Error": "Yes" if "parse_error" in llm_analysis else "No",
                    "Retry_Exhausted": "Yes" if llm_analysis.get("retry_exhausted") else "No",
                    "Summary": llm_analysis.get("summary", "")[:100] + "..." if len(
                        llm_analysis.get("summary", "")) > 100 else llm_analysis.get("summary", ""),
                    "Question_Preview": q["question_text"][:100] + "..." if len(q["question_text"]) > 100 else q[
                        "question_text"]
                }
                csv_data.append(row)

        if csv_data:
            csv_filename = f"{region_clean}_test_summary_{timestamp}.csv"
            csv_filepath = os.path.join(results_dir, csv_filename)

            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filepath, index=False, encoding='utf-8')

        print(f"\nRESULTS SAVED:")
        print(f"   Detailed results: {json_filepath}")
        if csv_data:
            print(f"   Summary CSV: {csv_filepath}")
        print(f"   Results directory: {os.path.abspath(results_dir)}")

    def run_full_region_analysis(self, json_directory: str, region_name: str, save_results: bool = True):
        """Execute full-scale analysis on all questions for a target region."""
        if self.client is None:
            print("[Error] Please set your actual OpenAI API key first!")
            return

        print(f"\n" + "=" * 80)
        print(f"FULL REGION ANALYSIS: {region_name.upper()}")
        print(f"=" * 80)
        print("[Warning] This will analyze ALL questions and cost money!")

        region_file = f"{region_name.replace(' ', '_').lower()}_llm_analysis.json"
        filepath = os.path.join(json_directory, region_file)

        if not os.path.exists(filepath):
            print(f"[Error] File not found: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data['questions_for_llm_analysis']
        total_questions = len(questions)

        print(f"About to analyze {total_questions} questions...")
        estimated_cost = 0
        for question in questions[:5]:
            prompt = self.create_prompt(question['question_text'], question['responses'], region_name)
            prompt_tokens = self.count_tokens(prompt)
            estimated_cost += self.calculate_cost(prompt_tokens, 400)

        full_estimated_cost = estimated_cost * (total_questions / 5)
        print(f"Estimated total cost: ${full_estimated_cost:.4f}")

        proceed = input(
            f"\n[Warning] Proceed with full analysis? This will cost approximately ${full_estimated_cost:.4f} (y/n): ")
        if proceed.lower() != 'y':
            print("Analysis cancelled.")
            return

        print(f"\nStarting full analysis of {total_questions} questions WITH RETRY LOGIC...")

        total_cost, results = self.run_actual_test(
            json_directory,
            region_name,
            num_questions=total_questions,
            save_results=save_results
        )

        print(f"\nFULL ANALYSIS COMPLETE!")
        print(f"Total cost: ${total_cost:.4f}")

        return results


def main():
    """
    Entry point for the analysis script.
    Configure target regions and analysis modes here.
    """

    analyzer = SimpleLLMAgreementAnalyzer()

    json_directory = "path to your directory"  # UPDATE THIS PATH

    # STEP 1: Cost estimation
    #print("STEP 1: Cost Estimation (No API calls, free)")
    #analyzer.test_cost_for_region(json_directory, "Central", num_questions=5)

    # STEP 2: Test with a few questions
    #print("\n" + "=" * 100)
    #print("STEP 2: Sample Test (costs small amount of money)")
    #cost, results = analyzer.run_actual_test(json_directory, "Central", num_questions=2, save_results=True)

    # STEP 3: Full region analysis
    print("\n" + "=" * 100)
    print("STEP 3: Full Region Analysis (costs more money)")
    full_results = analyzer.run_full_region_analysis(json_directory, "Central", save_results=True)


if __name__ == "__main__":
    main()