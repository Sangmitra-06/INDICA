"""
Cultural Dataset Annotation Tool
================================

A Streamlit-based interface for annotating cultural commonsense datasets. 
This tool facilitates the review and annotation of regional responses, 
agreement analysis, and metadata classification (Category/Subcategory/Topic).
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import os

# Taxonomy structure defining the hierarchical organization of cultural topics.
# Used for classifying questions into Category -> Subcategory -> Topic.
CATEGORY_HIERARCHY = {
    "Interpersonal Relations": {
        "Visiting and hospitality": [
            "Etiquette in the reception of visitors",
            "Occasions for visiting"
        ],
        "Gift giving": [
            "Gift Giving Etiquette",
            "Ceremonial gift giving"
        ],
        "Etiquette": [
            "Greeting and Salutation Etiquette",
            "Eating, drinking, and smoking etiquette"
        ]
    },
    "Festivals and Rituals": {
        "Rest days and holidays": [
            "Conceptualization of Holidays",
            "Secular Festival Practices",
            "Commemoration of Personal Milestones",
            "Religious Taboos on Holidays"
        ],
        "Ritual": [
            "Symbolic Act Performance",
            "Ritual Gestures",
            "Pilgrimage Practices"
        ],
        "Organized ceremonial": [
            "Engaging with Religious Music and Dance",
            "Timing of Ceremonies"
        ]
    },
    "Traffic and transport behavior": {
        "Streets and traffic": [
            "Understanding Local Traffic Regulations",
            "Adapting to Local Transportation Modes"
        ],
        "Transportation": [
            "Carrying Capacity of Transport",
            "Transport during Special Events"
        ]
    },
    "Education": {
        "Education system": [
            "Formal Educational Structure",
            "Attitudes toward Education"
        ],
        "Teachers and Students": [
            "Norms for Interacting with Teachers",
            "Student Extracurricular Activities"
        ]
    },
    "Clothing and adornment": {
        "Special garments": [
            "Special Occasion Clothing",
            "Headgear and Footwear Norms"
        ],
        "Ornament": [
            "Ornamental Attire and Status Indication",
            "Occasions for Wearing Specific Ornaments"
        ]
    },
    "Food processing and consumption": {
        "Food preparation": [
            "Food Preparation Techniques",
            "Cultural Recipes and Ingredients"
        ],
        "Diet": [
            "Staple Food Consumption",
            "Seasonal Diet Modifications"
        ]
    },
    "Communication": {
        "Gestures & Signs": [
            "Social Expression of Emotions",
            "Non-Verbal Expression of Respect or Disrespect"
        ],
        "Dissemination of News and Information": [
            "Navigating the \"Grapevine\"",
            "Trustworthiness of Information Sources"
        ]
    },
    "Finance": {
        "Credit": [
            "Negotiating Credit Advances and Discounts",
            "Navigating Installment Buying"
        ],
        "Saving & Investment": [
            "Safekeeping of Valuables",
            "Preferable Investment Forms"
        ]
    }
}


class AnnotationInterface:
    def __init__(self, data_file: str = "manual_annotation_helper.json"):
        self.data_file = data_file
        self.load_data()

    def load_data(self):
        """Load the consolidated annotation data from JSON."""
        # Attempt to resolve the dataset file path by checking common locations relative to execution context.
        possible_paths = [
            self.data_file,
            Path(self.data_file),
            Path(__file__).parent / self.data_file,
            Path.cwd() / self.data_file
        ]

        loaded = False
        for path in possible_paths:
            try:
                st.write(f"Trying path: {path}")  # Log path attempt for debugging
                with open(path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    self.data_file = str(path)  # Update to working path
                    loaded = True
                    st.success(f"✅ Loaded data from: {path}")
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
                continue

        if not loaded:
            st.error(f"❌ Could not find data file. Searched in:")
            for path in possible_paths:
                st.write(f"  - {path}")
            st.write(f"\nCurrent working directory: {os.getcwd()}")
            st.write(f"Files in current directory: {os.listdir('..')}")
            self.data = {"questions": []}

    def save_data(self):
        """Persist current annotations to the JSON file."""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def build_metadata_lookup(self, questions_file: str = "questions.json"):
        """
        Create a lookup dictionary mapping question text to its metadata (Category, Subcategory, Topic).
        Reads from the source questions file to enable auto-classification.
        """
        # Resolve path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, questions_file)

        if not os.path.exists(full_path):
            return {}

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return {}

        lookup = {}
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

        return lookup

    def auto_classify_all(self) -> int:
        """
        Batch classify all questions by matching their text against the metadata lookup.
        Updates empty metadata fields without overwriting existing manual annotations.
        """
        lookup = self.build_metadata_lookup()

        if not lookup:
            return 0

        count = 0
        for question in self.data.get("questions", []):
            q_text = question.get("question_text", "")
            metadata = lookup.get(q_text)

            if metadata:
                manual_space = question.get("manual_annotation_space", {})
                # Only update if not already set
                if not manual_space.get("question_category"):
                    manual_space["question_category"] = metadata.get("category", "")
                    manual_space["question_subcategory"] = metadata.get("subcategory", "")
                    manual_space["question_topic"] = metadata.get("topic", "")
                    question["manual_annotation_space"] = manual_space
                    count += 1

        if count > 0:
            self.save_data()

        return count

    def export_final_dataset(self, output_file: str = "annotated_dataset.json"):
        """
        Export verified annotations to a clean JSON structure.
        Filters for questions that have at least one regional answer annotated.
        """
        final_dataset = {"questions": []}

        for q in self.data["questions"]:
            manual_space = q.get("manual_annotation_space", {})

            # Determine if the question has been effectively annotated (contains at least one regional answer).
            has_answers = any([
                manual_space.get("final_answer_north"),
                manual_space.get("final_answer_south"),
                manual_space.get("final_answer_east"),
                manual_space.get("final_answer_west"),
                manual_space.get("final_answer_central")
            ])

            if has_answers:
                question_entry = {
                    "question_id": q["question_id"],
                    "question_text": q["question_text"],
                    "question_category": manual_space.get("question_category", ""),
                    "question_subcategory": manual_space.get("question_subcategory", ""),
                    "question_topic": manual_space.get("question_topic", "")
                }

                # Structure regional answers into the final output format.
                for region in ["North", "South", "East", "West", "Central"]:
                    region_key = f"final_answer_{region.lower()}"
                    answer = manual_space.get(region_key, [])
                    if answer:
                        question_entry[region] = {"answer": answer}

                # Include agreement data in the export.
                question_entry["pairwise_agreements"] = q["pairwise_agreements"]
                question_entry["universal_agreement"] = q["universal_agreement"]

                final_dataset["questions"].append(question_entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)

        return len(final_dataset["questions"])


def render_concept_display(concepts: List[Dict], region_name: str):
    """
    Render extracted cultural concepts and their associated evidence (quotes/notes).
    Used to display region-specific semantic concepts found in the data.
    """
    if not concepts:
        st.info(f"No concepts found for {region_name}")
        return

    for i, concept in enumerate(concepts):
        with st.expander(f"📌 {concept.get('concept', 'Unnamed concept')}"):
            st.markdown(f"**Semantic Note:** {concept.get('semantic_note', 'N/A')}")

            quotes = concept.get('exact_quotes_proof', [])
            if quotes:
                st.markdown("**Evidence Quotes:**")
                for quote in quotes[:5]:  # Show first 5
                    st.markdown(f"- *\"{quote}\"*")

            responses_count = len(concept.get('responses_mentioning', []))
            st.markdown(f"**Mentioned in {responses_count} responses**")


def render_raw_responses(responses: List[str], region_name: str):
    """Render raw participant responses for verification and context."""
    if not responses:
        st.info(f"No responses available for {region_name}")
        return

    with st.expander(f"👥 View Raw Responses ({len(responses)} total)"):
        for i, response in enumerate(responses, 1):
            st.markdown(f"**Response {i}:**")
            st.text(response)
            st.divider()


def render_pairwise_agreements(current_question: Dict):
    """
    Render an interactive editor for pairwise regional agreements.
    Handles state management to track and flag manual overrides to the agreement data.
    """
    st.subheader("🔗 Pairwise Regional Agreements")

    agreements = current_question.get("pairwise_agreements", {})
    if not agreements:
        st.info("No pairwise data available")
        return

    # Initialize session state to track original agreement values for change detection.
    if 'original_agreements' not in st.session_state:
        st.session_state.original_agreements = {}

    question_id = current_question['question_id']

    # Cache initial agreement states to detect differences between original data and user edits.
    if question_id not in st.session_state.original_agreements:
        st.session_state.original_agreements[question_id] = {}
        for pair, data in agreements.items():
            if isinstance(data, dict):
                st.session_state.original_agreements[question_id][pair] = {
                    "agreement": data.get("agreement", False) if data.get("agreement") is not None else False,
                    "summary": data.get("summary", "")
                }
            else:
                st.session_state.original_agreements[question_id][pair] = {
                    "agreement": data if data is not None else False,
                    "summary": ""
                }

    # Format agreement data for the Streamlit data_editor widget.
    editor_data = []
    for pair, data in agreements.items():
        # Handle format
        if isinstance(data, dict):
            agreement_status = data.get("agreement")
            summary = data.get("summary", "")
            is_manual = data.get("is_manual", False)
        else:
            agreement_status = data
            summary = ""
            is_manual = False

        # Ensure boolean consistency for missing agreement values.
        if agreement_status is None:
            agreement_status = False

        editor_data.append({
            "Pair": pair,
            "Agreement": agreement_status,
            "Summary": summary,
            "Manual Override": is_manual
        })

    df = pd.DataFrame(editor_data)

    # Configure Streamlit data_editor column properties and interactive widgets.
    column_config = {
        "Pair": st.column_config.TextColumn("Pair", disabled=True, width="small"),
        "Agreement": st.column_config.CheckboxColumn(
            "Agreement",
            help="Check if regions agree (from file or manual edit)",
            width="small"
        ),
        "Summary": st.column_config.TextColumn("Summary", width="large"),
        "Manual Override": st.column_config.CheckboxColumn(
            "Manual Override",
            help="Check this if you manually edited the agreement",
            width="small"
        )
    }

    st.info("ℹ️ Edit Agreement/Summary as needed. Check 'Manual Override' for any row you change.")

    # Display the interactive agreement editor.
    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config=column_config,
        width="stretch",
        key=f"editor_{current_question['question_id']}"
    )

    # Compare current editor state against cached values to identify and flag manual modifications.
    original_for_question = st.session_state.original_agreements[question_id]

    for index, row in edited_df.iterrows():
        pair = row["Pair"]
        new_agreement = row["Agreement"]
        new_summary = row["Summary"]
        manual_override = row["Manual Override"]

        # Retrieve baseline values for change detection.
        original = original_for_question.get(pair, {"agreement": False, "summary": ""})
        orig_agreement = original["agreement"]
        orig_summary = original["summary"]

        # Determine if user modifications deviate from the baseline.
        agreement_changed = new_agreement != orig_agreement
        summary_changed = new_summary != orig_summary

        # Persist changes to the data structure.
        if not isinstance(agreements[pair], dict):
            agreements[pair] = {}

        agreements[pair]["agreement"] = new_agreement
        agreements[pair]["summary"] = new_summary

        # Logic to enforce 'manual_override' flag on detected changes.
        # Marks as manual if explicitly checked or if values differ from baseline.
        if manual_override:
            agreements[pair]["is_manual"] = True
        elif agreement_changed or summary_changed:
            # Automatically flag unlabeled edits and warn the user.
            agreements[pair]["is_manual"] = True
            if not manual_override:
                st.warning(f"⚠️ Change detected in '{pair}' but Manual Override not checked. Auto-marking as manual.")

    # Commit updates to the primary question object.
    current_question["pairwise_agreements"] = agreements

    st.markdown("---")

        # Statistic Summary Calculation
    total = len(agreements)
    true_count = sum(1 for v in agreements.values() if
                     (isinstance(v, dict) and v.get("agreement") is True) or (not isinstance(v, dict) and v is True))
    false_count = sum(1 for v in agreements.values() if
                      (isinstance(v, dict) and v.get("agreement") is False) or (not isinstance(v, dict) and v is False))
    manual_count = sum(1 for v in agreements.values() if isinstance(v, dict) and v.get("is_manual", False))

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Agreements", f"{true_count}/{total}")
    col2.metric("❌ Disagreements", f"{false_count}/{total}")
    col3.metric("✏️ Manual Edits", f"{manual_count}/{total}")


def render_matched_concepts(matched_concepts: List[Dict]):
    """Render cross-regional matched concepts"""
    if not matched_concepts:
        st.info("No matched concepts across regions")
        return

    st.subheader("🌍 Concepts Matched Across Regions")

    for i, mc in enumerate(matched_concepts, 1):
        with st.expander(f"Concept {i}: {mc.get('unified_concept_name', 'Unnamed')}"):
            regions = mc.get('regions_sharing', [])
            st.markdown(f"**Shared by:** {', '.join(regions)}")
            st.markdown(f"**Explanation:** {mc.get('semantic_explanation', 'N/A')}")

            if 'regional_variations' in mc:
                st.markdown("**Regional Variations:**")
                for region, variation in mc['regional_variations'].items():
                    st.markdown(f"- **{region}:** {variation}")


def main():
    st.set_page_config(
        page_title="Cultural Dataset Annotation Tool",
        page_icon="📝",
        layout="wide"
    )

    # Initialize the AnnotationInterface singleton in session state.
    if 'annotation_interface' not in st.session_state:
        st.session_state.annotation_interface = AnnotationInterface()

    interface = st.session_state.annotation_interface

    # Application Header
    st.title("📝 Cultural Dataset Annotation Tool")
    st.markdown("---")

    # Sidebar: Navigation controls and progress statistics
    with st.sidebar:
        st.header("Navigation")

        total_questions = len(interface.data.get("questions", []))

        # Calculate and display progress metrics.
        annotated_count = sum(
            1 for q in interface.data.get("questions", [])
            if any([
                q.get("manual_annotation_space", {}).get("final_answer_north"),
                q.get("manual_annotation_space", {}).get("final_answer_south"),
                q.get("manual_annotation_space", {}).get("final_answer_east"),
                q.get("manual_annotation_space", {}).get("final_answer_west"),
                q.get("manual_annotation_space", {}).get("final_answer_central")
            ])
        )

        st.metric("Total Questions", total_questions)
        st.metric("Annotated", annotated_count)
        st.progress(annotated_count / total_questions if total_questions > 0 else 0)

        # Auto-classify button
        if st.button("🏷️ Auto-Classify All",
                     help="Automatically populate category, subcategory, and topic from final_questions.json"):
            count = interface.auto_classify_all()
            if count > 0:
                st.success(f"✅ Auto-classified {count} questions!")
                st.rerun()
            else:
                st.info("All questions already have metadata or no matching questions found.")

        st.markdown("---")

        # Question Selection and Filtering Interface
        st.subheader("Select Question")

        # Filtering Logic
        filter_option = st.radio(
            "Filter by:",
            ["All Questions", "With Universal Agreement", "Without Universal Agreement",
             "Annotated", "Not Annotated", "Has Regional Agreement"]
        )

        # Apply selected filters to the question list.
        filtered_questions = interface.data.get("questions", [])

        if filter_option == "With Universal Agreement":
            filtered_questions = [q for q in filtered_questions if q.get("universal_agreement")]
        elif filter_option == "Without Universal Agreement":
            filtered_questions = [q for q in filtered_questions if not q.get("universal_agreement")]
        elif filter_option == "Annotated":
            filtered_questions = [
                q for q in filtered_questions
                if any([
                    q.get("manual_annotation_space", {}).get("final_answer_north"),
                    q.get("manual_annotation_space", {}).get("final_answer_south"),
                    q.get("manual_annotation_space", {}).get("final_answer_east"),
                    q.get("manual_annotation_space", {}).get("final_answer_west"),
                    q.get("manual_annotation_space", {}).get("final_answer_central")
                ])
            ]
        elif filter_option == "Not Annotated":
            filtered_questions = [
                q for q in filtered_questions
                if not any([
                    q.get("manual_annotation_space", {}).get("final_answer_north"),
                    q.get("manual_annotation_space", {}).get("final_answer_south"),
                    q.get("manual_annotation_space", {}).get("final_answer_east"),
                    q.get("manual_annotation_space", {}).get("final_answer_west"),
                    q.get("manual_annotation_space", {}).get("final_answer_central")
                ])
            ]
        elif filter_option == "Has Regional Agreement":
            filtered_questions = [
                q for q in filtered_questions
                if any([
                    q.get("regional_data", {}).get(region, {}).get("has_agreement")
                    for region in ["North", "South", "East", "West", "Central"]
                ])
            ]

        st.info(f"Showing {len(filtered_questions)} questions")

        # Question selection
        if filtered_questions:
            # Initialize session state index if needed
            if 'current_question_index' not in st.session_state:
                st.session_state.current_question_index = 0

            # Ensure index is valid
            if st.session_state.current_question_index >= len(filtered_questions):
                st.session_state.current_question_index = 0

            # Navigation buttons
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Previous", width="stretch"):
                    st.session_state.current_question_index = (st.session_state.current_question_index - 1) % len(
                        filtered_questions)
                    st.rerun()
            with col_next:
                if st.button("Next ➡️", width="stretch"):
                    st.session_state.current_question_index = (st.session_state.current_question_index + 1) % len(
                        filtered_questions)
                    st.rerun()

            question_options = [
                f"{q['question_id']}: {q['question_text'][:50]}..."
                for q in filtered_questions
            ]

            # Sync selectbox with session state
            selected_idx = st.selectbox(
                "Choose question:",
                range(len(filtered_questions)),
                format_func=lambda x: question_options[x],
                index=st.session_state.current_question_index
            )

            # Update session state if selectbox changes
            if selected_idx != st.session_state.current_question_index:
                st.session_state.current_question_index = selected_idx
                st.rerun()

            current_question = filtered_questions[st.session_state.current_question_index]
        else:
            st.warning("No questions match the filter")
            current_question = None

        st.markdown("---")

        # Data Persistence Controls
        st.subheader("Export")
        if st.button("💾 Save Progress", width="stretch"):
            interface.save_data()
            st.success("Progress saved!")

        if st.button("📤 Export Final Dataset", width="stretch"):
            count = interface.export_final_dataset()
            st.success(f"Exported {count} annotated questions to final_dataset.json")

    # Main Viewing Area
    if current_question:
        # Header and Editing Interface for Query Text
        col_header, col_edit = st.columns([4, 1])
        with col_header:
            st.header(f"{current_question['question_id']}")
        with col_edit:
            edit_mode = st.toggle("Edit Mode", value=False)

        if edit_mode:
            new_question_text = st.text_area(
                "Edit Question Text",
                value=current_question['question_text'],
                height=100
            )
            if new_question_text != current_question['question_text']:
                current_question['question_text'] = new_question_text
                st.info("Text updated (remember to Save Annotations)")
        else:
            st.subheader(current_question['question_text'])

        # Visual indicator for universal agreement status across all regions.
        if current_question.get("universal_agreement"):
            st.success("✅ Universal Agreement Found Across All Regions")
        else:
            st.warning("⚠️ No Universal Agreement")

        if current_question.get("agreement_summary"):
            with st.expander("📊 Agreement Summary"):
                st.info(current_question["agreement_summary"])

        st.markdown("---")

        # Main Layout: Split Screen
        # Layout Configuration: Split view for simultaneous analysis and annotation.
        # Left Panel: Contextual data and analysis.
        # Right Panel: Annotation inputs.
        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            st.subheader("📚 Analysis & Context")

            # Tab interface for exploring Region-Specific Data
            region_tabs = st.tabs(["North", "South", "East", "West", "Central"])
            regions = ["North", "South", "East", "West", "Central"]

            for tab, region in zip(region_tabs, regions):
                with tab:
                    regional_data = current_question.get("regional_data", {}).get(region, {})

                    # Agreement status
                    if regional_data.get("has_agreement"):
                        st.success(f"✅ {region} has intra-regional agreement")
                    else:
                        st.error(f"❌ {region} has NO intra-regional agreement")

                    # Suggested Answer (Prominent)
                    suggested = regional_data.get("suggested_answer", [])
                    if suggested:
                        st.info(f"💡 Suggested: {', '.join(suggested)}")

                    st.subheader("� Concepts")
                    concepts = regional_data.get("concepts", [])
                    render_concept_display(concepts, region)

                    st.subheader("💬 Responses")
                    responses = regional_data.get("raw_responses", [])
                    render_raw_responses(responses, region)

                    if regional_data.get("summary"):
                        st.markdown("**LLM Summary:**")
                        st.caption(regional_data["summary"])

            st.markdown("---")
            # Interactive pairwise agreement editor.
            render_pairwise_agreements(current_question)
            render_matched_concepts(current_question.get("matched_concepts_across_regions", []))

        with right_col:
            st.subheader("✍️ Annotation Workspace")

            manual_space = current_question.get("manual_annotation_space", {})

            # 1. Metadata Input: Cascading selection for hierarchical classification.
            with st.expander("Metadata", expanded=True):
                # Category
                current_cat = manual_space.get("question_category", "")
                cat_options = list(CATEGORY_HIERARCHY.keys())
                # Add current if not in list (preserve data)
                if current_cat and current_cat not in cat_options:
                    cat_options.append(current_cat)

                cat_index = cat_options.index(current_cat) if current_cat in cat_options else 0

                category = st.selectbox(
                    "Category",
                    options=cat_options,
                    index=cat_index,
                    key=f"category_{current_question['question_id']}"
                )

                # Subcategory (Cascading)
                current_sub = manual_space.get("question_subcategory", "")
                sub_options = list(CATEGORY_HIERARCHY.get(category, {}).keys())

                # Accommodate custom categories not present in the standard hierarchy.
                if not sub_options and category not in CATEGORY_HIERARCHY:
                    # Fallback to free text or empty list?
                    # If custom category, maybe allow custom subcategory?
                    # For now, let's treat it as empty list unless current_sub exists
                    pass

                if current_sub and current_sub not in sub_options:
                    sub_options.append(current_sub)

                # If options are empty (e.g. new category selected), default to first if available
                sub_index = sub_options.index(current_sub) if current_sub in sub_options else 0

                subcategory = st.selectbox(
                    "Subcategory",
                    options=sub_options,
                    index=sub_index if sub_options else None,
                    key=f"subcategory_{current_question['question_id']}",
                    disabled=not sub_options
                )

                # Topic (Cascading)
                current_topic = manual_space.get("question_topic", "")
                topic_options = CATEGORY_HIERARCHY.get(category, {}).get(subcategory, [])

                if current_topic and current_topic not in topic_options:
                    topic_options.append(current_topic)

                topic_index = topic_options.index(current_topic) if current_topic in topic_options else 0

                topic = st.selectbox(
                    "Topic",
                    options=topic_options,
                    index=topic_index if topic_options else None,
                    key=f"topic_{current_question['question_id']}",
                    disabled=not topic_options
                )

            # 2. Regional Final Answers: Input formulation of region-specific consensus.
            st.markdown("### Regional Answers")
            regional_answers = {}

            for region in regions:
                region_lower = region.lower()
                current_answer = manual_space.get(f"final_answer_{region_lower}", [])
                current_answer_str = "\n".join(current_answer) if current_answer else ""

                # Helper mechanism to copy suggested answers into the final answer field.
                reg_data = current_question.get("regional_data", {}).get(region, {})
                suggested_opts = reg_data.get("suggested_answer", [])
                suggested_str = "\n".join(suggested_opts) if suggested_opts else ""

                st.markdown(f"**{region}**")

                # Copy button
                if suggested_str:
                    if st.button(f"Copy Suggested to {region}", key=f"copy_{region}_{current_question['question_id']}"):
                        # Update the underlying data directly
                        manual_space[f"final_answer_{region_lower}"] = suggested_opts
                        current_question["manual_annotation_space"] = manual_space

                        # FORCE update the session state for the text area
                        text_area_key = f"answer_{region_lower}_{current_question['question_id']}"
                        st.session_state[text_area_key] = suggested_str

                        st.rerun()

                answer_text = st.text_area(
                    f"{region} Answer",
                    value=current_answer_str,
                    key=f"answer_{region_lower}_{current_question['question_id']}",
                    height=100,
                    label_visibility="collapsed",
                    placeholder=f"Enter {region} answer..."
                )

                regional_answers[f"final_answer_{region_lower}"] = [
                    line.strip() for line in answer_text.split("\n") if line.strip()
                ]

            # 3. Notes
            st.markdown("### Notes")
            notes = st.text_area(
                "Notes",
                value=manual_space.get("notes", ""),
                key=f"notes_{current_question['question_id']}",
                height=100
            )

            # 4. Save Controls
            st.markdown("---")
            if st.button("💾 Save All Changes", type="primary", width="stretch"):
                # Update the data
                manual_space.update({
                    "question_category": category,
                    "question_subcategory": subcategory,
                    "question_topic": topic,
                    "notes": notes,
                    **regional_answers
                })
                current_question["manual_annotation_space"] = manual_space
                interface.save_data()
                st.success("Saved!")

    else:
        st.info("👈 Select a question from the sidebar to begin annotation")


if __name__ == "__main__":
    main()