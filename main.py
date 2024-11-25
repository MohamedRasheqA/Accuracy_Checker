import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"])

# Evaluation criteria and steps
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Metric definitions
METRICS = {
    "Relevance": {
        "criteria": """Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.""",
        "steps": """1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5."""
    },
    "Coherence": {
        "criteria": """Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a \
coherent body of information about a topic.\"""",
        "steps": """1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."""
    },
    "Consistency": {
        "criteria": """Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.""",
        "steps": """1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria."""
    },
    "Fluency": {
        "criteria": """Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.""",
        "steps": """Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3."""
    }
}

def get_geval_score(criteria: str, steps: str, document: str, summary: str, metric_name: str):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

def evaluate_summaries(source_text, summaries):
    data = {"Evaluation Type": [], "Summary Type": [], "Score": []}
    
    for eval_type, metric_info in METRICS.items():
        for summ_type, summary in summaries.items():
            data["Evaluation Type"].append(eval_type)
            data["Summary Type"].append(summ_type)
            result = get_geval_score(
                metric_info["criteria"],
                metric_info["steps"],
                source_text,
                summary,
                eval_type
            )
            score_num = int(result.strip())
            data["Score"].append(score_num)
    
    return pd.DataFrame(data)

# Streamlit UI
st.title("Summary Evaluation App")

# Source text input
st.header("Source Text")
source_text = st.text_area(
    "Enter the source text to be summarized:",
    height=200,
    help="Paste the original text here that you want to evaluate summaries against."
)

# Summary inputs
st.header("Summaries")
num_summaries = st.number_input("Number of summaries to evaluate", min_value=1, max_value=5, value=2)

summaries = {}
for i in range(num_summaries):
    summary = st.text_area(
        f"Summary {i+1}:",
        height=100,
        key=f"summary_{i}",
        help=f"Enter summary {i+1} here"
    )
    if summary:
        summaries[f"Summary {i+1}"] = summary

# Evaluate button
if st.button("Evaluate Summaries") and source_text and all(summaries.values()):
    st.header("Evaluation Results")
    
    with st.spinner("Evaluating summaries..."):
        # Get evaluation results
        results_df = evaluate_summaries(source_text, summaries)
        
        # Create pivot table
        pivot_df = results_df.pivot(
            index="Evaluation Type",
            columns="Summary Type",
            values="Score"
        )
        
        # Display results
        try:
            st.dataframe(
                pivot_df.style.background_gradient(
                    cmap='RdYlGn',
                    axis=1,
                    vmin=1,
                    vmax=5
                )
            )
        except ImportError:
            st.error("Matplotlib is required for displaying the gradient. Please install it using 'pip install matplotlib'.")
        
        # Display detailed breakdown
        st.subheader("Detailed Scores")
        for eval_type in METRICS.keys():
            st.write(f"**{eval_type}**")
            type_scores = pivot_df.loc[eval_type]
            
            # Create a bar chart for this metric
            st.bar_chart(type_scores)
            
            # Show the criteria used
            with st.expander(f"View {eval_type} Criteria"):
                st.write(METRICS[eval_type]["criteria"])
else:
    st.info("Please enter the source text and at least one summary to begin evaluation.")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use this app:
1. Paste your source text in the first text box
2. Enter the number of summaries you want to evaluate
3. Paste each summary in the provided text boxes
4. Click 'Evaluate Summaries' to see the results
""")
