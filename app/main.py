import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.utils import run_newsletter_workflow

st.set_page_config(page_title="ContentWeaver AI", page_icon="📰", layout="wide")
st.title("ContentWeaver AI")
st.markdown("Get a personalized newsletter based on your interests!")

st.sidebar.header("Preferences")
keywords_input = st.sidebar.text_area(
    "Enter topics/keywords (comma-separated):",
    "AI agents, LLM applications, RAG technology",
)
selected_tone = st.sidebar.selectbox(
    "Select newsletter tone:", ["Informative", "Enthsiastic"], index=0
)
craft_button = st.sidebar.button("Craft")

if craft_button:
    if not keywords_input.strip():
        st.error("Please enter some keywords!")
    else:
        preferences_dict = {
            "keywords": [kw.strip() for kw in keywords_input.split(",")],
            "preferred_tone": selected_tone,
        }

        with st.spinner("Finding articsles, thinking, writing....please wait."):
            try:
                newsletter_markdown, status_message = run_newsletter_workflow(
                    preferences_dict
                )

                if newsletter_markdown:
                    st.success(status_message or "Newsletter ready!")
                    st.markdown("---")
                    st.subheader("Your personalized digest:")
                    st.markdown(newsletter_markdown, unsafe_allow_html=True)
                else:
                    st.warning(
                        status_message
                        or "Could not generate newsletter. Try again later."
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Adjust your preferences in the sidebar and click 'Craft'")
