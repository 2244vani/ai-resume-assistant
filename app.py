import streamlit as st
from typing import TypedDict
import re

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄"
)

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
        f.write(uploaded_file.getbuffer())


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


    st.success("Resume processed successfully!")

    st.write(
        "Number of resume chunks:",
        len(chunks)
    )


    # ======================================================
    # EMBEDDINGS
    # ======================================================

    @st.cache_resource
    def get_embeddings():

        return OllamaEmbeddings(
            model="nomic-embed-text"
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
    # BUTTON
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
        # OLLAMA MODEL
        # ==================================================

        model = ChatOllama(
            model="llama3.2"
        )


        # ==================================================
        # NODE 1
        # RESUME RETRIEVAL
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
        # RESUME ANALYSIS
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


Give the answer using EXACTLY these sections:

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

Do not include preferred skills.

Do not include education.

Do not include experience.


3. PREFERRED SKILLS

List ONLY the skills under the
"Preferred Skills" section of the job description.

Do not include required skills.


4. MATCHING SKILLS

List ONLY the REQUIRED SKILLS that clearly
appear in the resume.

Do not include preferred skills.

Do not guess.


5. MISSING SKILLS

List ONLY the REQUIRED SKILLS that are not
found in the resume.

Do not include preferred skills.

Do not include skills that are already matching.


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
"""


            response = model.invoke(
                prompt
            )


            answer = response.content


            # ==================================================
            # FIND SECTIONS
            # ==================================================

            required_start = answer.find(
                "2. REQUIRED SKILLS"
            )

            preferred_start = answer.find(
                "3. PREFERRED SKILLS"
            )

            matching_start = answer.find(
                "4. MATCHING SKILLS"
            )

            missing_start = answer.find(
                "5. MISSING SKILLS"
            )


            # ==================================================
            # CALCULATE MATCH PERCENTAGE
            # ==================================================

            match_percentage = 0


            if (
                required_start != -1
                and matching_start != -1
                and missing_start != -1
            ):

                required_section = answer[
                    required_start:preferred_start
                    if preferred_start != -1
                    else matching_start
                ]


                matching_section = answer[
                    matching_start:missing_start
                ]


                # ----------------------------------------------
                # REQUIRED SKILLS
                # ----------------------------------------------

                required_skills = []

                for line in required_section.splitlines():

                    line = line.strip()

                    if line.startswith("-"):

                        skill = line[1:].strip()

                        if skill:

                            required_skills.append(
                                skill
                            )


                # Remove duplicates

                unique_required = []

                seen = set()


                for skill in required_skills:

                    key = skill.lower()

                    if key not in seen:

                        seen.add(key)

                        unique_required.append(
                            skill
                        )


                # ----------------------------------------------
                # MATCHING SKILLS
                # ----------------------------------------------

                matching_skills = []

                for line in matching_section.splitlines():

                    line = line.strip()

                    if line.startswith("-"):

                        skill = line[1:].strip()

                        if skill:

                            matching_skills.append(
                                skill
                            )


                # Remove duplicates

                unique_matching = []

                seen = set()


                for skill in matching_skills:

                    key = skill.lower()

                    if key not in seen:

                        seen.add(key)

                        unique_matching.append(
                            skill
                        )


                # ----------------------------------------------
                # VALID MATCHES
                # ----------------------------------------------

                valid_matches = []


                for matching_skill in unique_matching:

                    for required_skill in unique_required:

                        if (
                            matching_skill.lower()
                            == required_skill.lower()
                        ):

                            if required_skill not in valid_matches:

                                valid_matches.append(
                                    required_skill
                                )


                # ----------------------------------------------
                # PERCENTAGE
                # ----------------------------------------------

                if unique_required:

                    match_percentage = (
                        len(valid_matches)
                        /
                        len(unique_required)
                    ) * 100


                match_percentage = round(
                    match_percentage,
                    2
                )


            # ==================================================
            # REPLACE PERCENTAGE
            # ==================================================

            answer = re.sub(
                r"6\.\s*MATCH PERCENTAGE:?.*?(?=7\.\s*SUGGESTIONS:)",
                f"6. MATCH PERCENTAGE:\n\n"
                f"{match_percentage}%",
                answer,
                flags=re.DOTALL
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
        # RUN
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
        # MATCH PERCENTAGE
        # ==================================================

        percentage_match = re.search(
            r"6\.\s*MATCH PERCENTAGE:?\s*(\d+(?:\.\d+)?)%",
            answer
        )


        if percentage_match:

            match_percentage = float(
                percentage_match.group(1)
            )

            st.metric(
                "Resume Match Percentage",
                f"{match_percentage}%"
            )


        # ==================================================
        # COMPLETE ANSWER
        # ==================================================

        st.markdown(
            answer
        )