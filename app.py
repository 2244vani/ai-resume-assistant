import os
import re
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


# ==========================================================
# GET GROQ API KEY
# ==========================================================

def get_groq_api_key():

    # Streamlit Cloud
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Local .env
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


if not GROQ_API_KEY:

    st.error("GROQ_API_KEY is not configured.")

    st.info(
        "Add GROQ_API_KEY to your local .env file "
        "or Streamlit Cloud Secrets."
    )

    st.stop()


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📄 AI Resume Assistant")

st.write(
    "Upload your resume and analyze it against a job description."
)


# ==========================================================
# UPLOAD RESUME
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success("Resume uploaded successfully!")


    # ======================================================
    # SAVE PDF
    # ======================================================

    with open("uploaded_resume.pdf", "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # ======================================================
    # LOAD PDF
    # ======================================================

    loader = PyPDFLoader(
        "uploaded_resume.pdf"
    )

    documents = loader.load()


    # ======================================================
    # SPLIT DOCUMENT
    # ======================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(
        documents
    )


    st.success(
        "Resume processed successfully!"
    )

    st.write(
        "Number of resume chunks:",
        len(chunks)
    )


    # ======================================================
    # HUGGING FACE EMBEDDINGS
    # ======================================================

    @st.cache_resource
    def get_embeddings():

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


    embeddings = get_embeddings()


    # ======================================================
    # CHROMA VECTOR DATABASE
    # ======================================================

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="resume_app"
    )


    st.success(
        "Resume is ready for questions!"
    )


    # ======================================================
    # JOB DESCRIPTION
    # ======================================================

    job_description = st.text_area(
        "Paste the Job Description",
        height=250
    )


    # ======================================================
    # QUESTION
    # ======================================================

    question = st.text_input(
        "Ask a question about your resume and job description",
        value="Analyze my resume against this job description."
    )


    # ======================================================
    # ANALYZE BUTTON
    # ======================================================

    analyze_button = st.button(
        "🔍 Analyze Resume"
    )


    if analyze_button:

        if not job_description.strip():

            st.warning(
                "Please paste a job description."
            )

            st.stop()


        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()


        # ==================================================
        # STATE
        # ==================================================

        class State(TypedDict):

            question: str
            job_description: str
            context: str
            answer: str


        # ==================================================
        # GROQ MODEL
        # ==================================================

        model = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=GROQ_API_KEY
        )


        # ==================================================
        # NODE 1
        # RETRIEVE RESUME
        # ==================================================

        def retrieve_resume(
            state: State
        ):

            results = vector_store.similarity_search(
                state["question"],
                k=min(7, len(chunks))
            )


            context = "\n\n".join(
                result.page_content
                for result in results
            )


            return {
                "context": context
            }


        # ==================================================
        # NODE 2
        # ANALYZE RESUME
        # ==================================================

        def analyze_resume(
            state: State
        ):

            prompt = f"""
You are an AI Resume Assistant.

Analyze the candidate's resume against the job description.

==============================
RESUME
==============================

{state["context"]}


==============================
JOB DESCRIPTION
==============================

{state["job_description"]}


Return the answer using EXACTLY these sections:

1. RESUME SUMMARY

Give a short summary of:

- Education
- Skills
- Projects
- Experience

Use ONLY information found in the resume.

Do not invent anything.


2. REQUIRED SKILLS

List ONLY the skills under the
"Required Skills" section of the job description.

Write each skill on a separate line.

Do not include preferred skills.

Do not write explanations.


3. PREFERRED SKILLS

List ONLY the skills under the
"Preferred Skills" section of the job description.

Write each skill on a separate line.

Do not include required skills.

Do not write explanations.


4. MATCHING SKILLS

List ONLY the REQUIRED SKILLS that clearly
appear in the resume.

Write each skill on a separate line.

Do not include preferred skills.

Do not guess.

Do not write explanations.


5. MISSING SKILLS

List ONLY the REQUIRED SKILLS that are not
found in the resume.

Write each skill on a separate line.

Do not include preferred skills.

Do not include skills that are already matching.

Do not write explanations.


6. MATCH PERCENTAGE

Write exactly:

Match percentage will be calculated separately.


7. SUGGESTIONS

Give practical suggestions for improving the
resume for this job.

Do not invent experience.

Do not falsely claim that the candidate has
a skill.


IMPORTANT RULES:

- Do not invent information.
- Do not duplicate skills.
- Keep required skills separate from preferred skills.
- Matching skills must come only from required skills.
- Missing skills must come only from required skills.
- Use simple language.
- Put every skill on its own line.
"""


            # ==================================================
            # CALL GROQ
            # ==================================================

            try:

                response = model.invoke(
                    prompt
                )

                answer = response.content

            except Exception as e:

                return {
                    "answer": f"ERROR: {str(e)}"
                }


            # ==================================================
            # CLEAN MARKDOWN FORMATTING
            # ==================================================

            # Convert escaped markdown
            answer = answer.replace(
                r"\*\*",
                "**"
            )

            # Remove bold/italic formatting specifically
            # around numbered section headings.
            section_names = [
                "RESUME SUMMARY",
                "REQUIRED SKILLS",
                "PREFERRED SKILLS",
                "MATCHING SKILLS",
                "MISSING SKILLS",
                "MATCH PERCENTAGE",
                "SUGGESTIONS"
            ]

            for number, section_name in enumerate(
                section_names,
                start=1
            ):

                pattern = (
                    r"\*{0,3}"
                    r"\s*"
                    + str(number)
                    + r"\.\s*"
                    + re.escape(section_name)
                    + r"\s*"
                    r"\*{0,3}"
                )

                answer = re.sub(
                    pattern,
                    f"{number}. {section_name}",
                    answer,
                    flags=re.IGNORECASE
                )


            # ==================================================
            # FIND SECTION POSITIONS
            # ==================================================

            required_match = re.search(
                r"(?im)^\s*2\.\s*REQUIRED SKILLS\s*$",
                answer
            )

            preferred_match = re.search(
                r"(?im)^\s*3\.\s*PREFERRED SKILLS\s*$",
                answer
            )

            matching_match = re.search(
                r"(?im)^\s*4\.\s*MATCHING SKILLS\s*$",
                answer
            )

            missing_match = re.search(
                r"(?im)^\s*5\.\s*MISSING SKILLS\s*$",
                answer
            )

            percentage_match = re.search(
                r"(?im)^\s*6\.\s*MATCH PERCENTAGE\s*$",
                answer
            )

            suggestions_match = re.search(
                r"(?im)^\s*7\.\s*SUGGESTIONS\s*$",
                answer
            )


            # ==================================================
            # EXTRACT SKILLS FROM A SECTION
            # ==================================================

            def extract_skills(section_text):

                skills = []

                for line in section_text.splitlines():

                    line = line.strip()

                    if not line:
                        continue


                    # Remove markdown bullets
                    line = re.sub(
                        r"^[-•*]\s*",
                        "",
                        line
                    )


                    # Remove markdown formatting
                    line = line.replace(
                        "**",
                        ""
                    )

                    line = line.replace(
                        "__",
                        ""
                    )

                    line = line.replace(
                        "`",
                        ""
                    )

                    line = line.strip()


                    # Ignore headings
                    if re.match(
                        r"^\d+\.",
                        line
                    ):
                        continue


                    # Ignore instruction-like text
                    lower_line = line.lower()

                    if lower_line.startswith(
                        (
                            "list only",
                            "do not",
                            "write each",
                            "match percentage will",
                            "none",
                            "n/a"
                        )
                    ):
                        continue


                    skills.append(line)


                # Remove duplicates
                unique_skills = []

                seen = set()

                for skill in skills:

                    key = skill.lower().strip()

                    if key not in seen:

                        seen.add(key)

                        unique_skills.append(
                            skill
                        )

                return unique_skills


            # ==================================================
            # CALCULATE MATCH PERCENTAGE
            # ==================================================

            match_percentage_value = 0


            if (
                required_match
                and preferred_match
                and matching_match
                and missing_match
            ):

                # ----------------------------------------------
                # REQUIRED SKILLS
                # ----------------------------------------------

                required_section = answer[
                    required_match.end():
                    preferred_match.start()
                ]


                required_skills = extract_skills(
                    required_section
                )


                # ----------------------------------------------
                # MATCHING SKILLS
                # ----------------------------------------------

                matching_section = answer[
                    matching_match.end():
                    missing_match.start()
                ]


                matching_skills = extract_skills(
                    matching_section
                )


                # ----------------------------------------------
                # VALID MATCHES
                # ----------------------------------------------

                valid_matches = []


                for matching_skill in matching_skills:

                    for required_skill in required_skills:

                        if (
                            matching_skill.strip().lower()
                            ==
                            required_skill.strip().lower()
                        ):

                            if required_skill not in valid_matches:

                                valid_matches.append(
                                    required_skill
                                )


                # ----------------------------------------------
                # CALCULATE
                # ----------------------------------------------

                if required_skills:

                    match_percentage_value = round(
                        (
                            len(valid_matches)
                            /
                            len(required_skills)
                        ) * 100,
                        2
                    )


            # ==================================================
            # REPLACE MATCH PERCENTAGE SECTION
            # ==================================================

            if (
                percentage_match
                and suggestions_match
            ):

                before_percentage = answer[
                    :percentage_match.start()
                ]

                after_percentage = answer[
                    suggestions_match.start():
                ]

                answer = (
                    before_percentage
                    + "6. MATCH PERCENTAGE\n\n"
                    + f"{match_percentage_value}%"
                    + "\n\n"
                    + after_percentage
                )


            return {
                "answer": answer
            }


        # ==================================================
        # CREATE LANGGRAPH
        # ==================================================

        graph = StateGraph(
            State
        )


        # ==================================================
        # ADD NODES
        # ==================================================

        graph.add_node(
            "retrieve_resume",
            retrieve_resume
        )

        graph.add_node(
            "analyze_resume",
            analyze_resume
        )


        # ==================================================
        # EDGES
        # ==================================================

        graph.add_edge(
            START,
            "retrieve_resume"
        )

        graph.add_edge(
            "retrieve_resume",
            "analyze_resume"
        )

        graph.add_edge(
            "analyze_resume",
            END
        )


        # ==================================================
        # COMPILE
        # ==================================================

        app = graph.compile()


        # ==================================================
        # RUN LANGGRAPH
        # ==================================================

        with st.spinner(
            "AI is analyzing your resume..."
        ):

            result = app.invoke(
                {
                    "question": question,
                    "job_description": job_description,
                    "context": "",
                    "answer": ""
                }
            )


        # ==================================================
        # RESULT
        # ==================================================

        answer = result["answer"]


        st.subheader(
            "📊 Resume Analysis"
        )


        # ==================================================
        # GET MATCH PERCENTAGE
        # ==================================================

        final_percentage = re.search(
            r"6\.\s*MATCH PERCENTAGE\s+(\d+(?:\.\d+)?)%",
            answer,
            flags=re.IGNORECASE
        )


        if final_percentage:

            match_percentage = float(
                final_percentage.group(1)
            )

            st.metric(
                "Resume Match Percentage",
                f"{match_percentage:g}%"
            )


        # ==================================================
        # DISPLAY COMPLETE ANSWER
        # ==================================================

        st.markdown(
            answer
        )