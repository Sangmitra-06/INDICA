# 🇮🇳 INDICA

This is the repository for [Common to _Whom_? Regional Cultural Commonsense and LLM Bias in India](https://arxiv.org/abs/2510.20543). 

**📦 Dataset:** Available on [Hugging Face](https://huggingface.co/datasets/Sangmitra-06/CENTERBENCH)

Authors: Sangmitra Madhusudan, Trush Shashank More, Steph Buongiorno, Renata Dividino, Jad Kabbara and Ali Emami

## 📄 Paper abstract

Existing cultural commonsense benchmarks treat nations as _monolithic_, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce **INDICA**, the first benchmark designed to test LLMs' ability to address this question, focusing on India—a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly _regional_, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%–20.9% accuracy on region-specific questions, and they exhibit  geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement.
