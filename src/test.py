import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline


# Initialize the models
@st.cache_resource
def load_models():
    # Question Generation
    qg_model_name = "facebook/bart-large"
    qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
    qg_pipeline = pipeline("text2text-generation", model=qg_model, tokenizer=qg_tokenizer)

    # Question Answering
    qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

    return qg_pipeline, qa_pipeline


qg_pipeline, qa_pipeline = load_models()


# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text


# Function to generate questions and answers
def generate_qa(text, max_questions=5):
    qa_pairs = []
    sentences = text.split('. ')

    for i, sentence in enumerate(sentences):
        if i >= max_questions:
            break

        try:
            # Generate questions
            question_input = "Generate a question about: " + sentence
            questions = qg_pipeline(question_input)
            for q in questions:
                question = q['generated_text']
                # Generate answers
                answer_input = {"question": question, "context": text}
                answer = qa_pipeline(answer_input)
                qa_pairs.append({"question": question, "answer": answer['answer']})
        except Exception as e:
            st.error(f"An error occurred: {e}")
            continue

    return qa_pairs


# Streamlit app
st.title("PDF Flashcard Generator")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Generating questions and answers..."):
        flashcards = generate_qa(pdf_text)

    if flashcards:
        st.success("Flashcards generated successfully!")
        st.session_state.flashcards = flashcards

        # Initialize session state variables
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0

        if 'show_answer' not in st.session_state:
            st.session_state.show_answer = False


        # Function to go to the next flashcard
        def next_flashcard():
            st.session_state.current_index = (st.session_state.current_index + 1) % len(st.session_state.flashcards)
            st.session_state.show_answer = False


        # Function to show the answer
        def show_answer():
            st.session_state.show_answer = True


        # Display the current flashcard
        current_flashcard = st.session_state.flashcards[st.session_state.current_index]

        st.write(f"Question: {current_flashcard['question']}")

        if st.session_state.show_answer:
            st.write(f"Answer: {current_flashcard['answer']}")

        # Buttons for user interaction
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Show Answer"):
                show_answer()

        with col2:
            if st.button("Next Flashcard"):
                next_flashcard()
    else:
        st.error("No flashcards generated. Please try again with a different PDF.")
