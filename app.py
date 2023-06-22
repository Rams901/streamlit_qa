from typing import Dict
import numpy as np
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain import PromptTemplate
import pandas as pd
from langchain.vectorstores import FAISS
import requests
from typing import List
from langchain.document_loaders import YoutubeLoader
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
import pandas as pd
import PyPDF2
import streamlit as st

from io import BytesIO

import os

st.write(
    "Has environment variables been set:",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_KEY"],
)
st.title("Question-Answering RentGPT")


@st.cache(allow_output_mutation=True)
def extract_text_from_pdfs(pdf_files):

    # Create an empty data frame
    df = pd.DataFrame(columns=["file", "text"])
    # Iterate over the PDF files
    for pdf_file in pdf_files:
        # Open the PDF file
        # with open(pdf_file.read(), "rb") as f:
        with BytesIO(pdf_file.read()) as f:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(f)
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            # Initialize a string to store the text from the PDF
            text = ""

            # Iterate over all the pages
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract the text from the page
                page_text = page.extract_text()
                # Add the page text to the overall text
                text += page_text
            # Add the file name and the text to the data frame
            df = pd.concat((df, pd.DataFrame([{"file": pdf_file.name, "text": text}])), ignore_index = True)
    # Return the data frame
    return df


def preprocess_text(text_list):
    # Initialize a empty list to store the pre-processed text
    processed_text = []
    # Iterate over the text in the list
    for text in text_list:
        num_words = len(text.split(" "))
        if num_words > 10:  # only include sentences with length >10
            processed_text.append(text)
    # Return the pre-processed text
    return processed_text


def remove_short_sentences(df):
    df["sentences"] = df["sentences"].apply(preprocess_text)
    return df


# @st.cache(allow_output_mutation=True)
# def get_relevant_texts(df, topic):
    
#     model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
#     model_embedding.save("all-MiniLM-L6-v2")
#     cosine_threshold = 0.3  # set threshold for cosine similarity value
#     queries = topic  # search query
#     results = []
    
#     for i, document in enumerate(df["sentences"]):

#         sentence_embeddings = model_embedding.encode(document)
#         query_embedding = model_embedding.encode(queries)
#         for j, sentence_embedding in enumerate(sentence_embeddings):

#             distance = cosine_similarity(
#                 sentence_embedding.reshape((1, -1)), query_embedding.reshape((1, -1))
#             )[0][0]

#             sentence = df["sentences"].iloc[i][j]
#             results += [(i, sentence, distance)]
    
#     results = sorted(results, key=lambda x: x[2], reverse=True)
#     del model_embedding

#     texts = []
#     for idx, sentence, distance in results:
#         if distance > cosine_threshold:
#             text = sentence
#             texts.append(text)
#     # turn the list to string

#     context = "".join(texts)
#     return context


@st.cache(allow_output_mutation=True)
def get_pipeline():
    # modelname = "deepset/bert-base-cased-squad2"
    # model_qa = BertForQuestionAnswering.from_pretrained(modelname)
    # # model_qa.save_pretrained(modelname)
    # tokenizer = AutoTokenizer.from_pretrained("tokenizer-deepset")
    # # tokenizer.save_pretrained("tokenizer-" + modelname)
    # qa = pipeline("question-answering", model=model_qa, tokenizer=tokenizer)

    llm = ChatOpenAI(
            temperature=0,
            model='gpt-3.5-turbo',
        openai_api_key = st.secrets('OPENAI_KEY')
        )
    prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            As a consultant, your role is to assist the user in analyzing your mortgage documents and providing advice based on the information the user provides.
            You will help the user with questions related to your insurance, property tax, trends, and cost rates using the documents you provide.
            Please share your mortgage documents, including the mortgage agreement, insurance policies, and any other relevant paperwork.
            Once you have reviewed the documents, you will be able to offer detailed analysis and guidance.
            Answer the following question: {question}
            Use the following documents: {docs}
            Only use the factual information from the documents to answer the question.
            If you feel like you don't have enough information to answer the question, say "I don't know".
            """,
        )


    # llm = BardLLM()
    chain = LLMChain(llm=llm, prompt = prompt)

    return chain

@st.cache(allow_output_mutation=True)
def answer_question(question: str, context: str): -> str
        llm = ChatOpenAI(
            temperature=0,
            model='gpt-3.5-turbo',
        openai_api_key = st.secrets('OPENAI_KEY')
        )
        prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            As a consultant, your role is to assist the user in analyzing your mortgage documents and providing advice based on the information the user provides.
            You will help the user with questions related to your insurance, property tax, trends, and cost rates using the documents you provide.
            Please share your mortgage documents, including the mortgage agreement, insurance policies, and any other relevant paperwork.
            Once you have reviewed the documents, you will be able to offer detailed analysis and guidance.
            Answer the following question: {question}
            Use the following documents: {docs}
            Only use the factual information from the documents to answer the question.
            If you feel like you don't have enough information to answer the question, say "I don't know".
            """,
        )


        # llm = BardLLM()
        chain = LLMChain(llm=llm, prompt = prompt)
        # input = {"question": question, "context": context}
        response = chain.run(question = question, docs = context)
        return str(response)


@st.cache(allow_output_mutation=True)
def create_context(df):
    # path = "data/"
    # files = Path(path).glob("WHR+22.pdf")
    # df = extract_text_from_pdfs(files)
    embeddings = HuggingFaceEmbeddings()

    df["sentences"] = df["text"].apply(
        lambda long_str: long_str.replace("\n", " ").split(".")
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.create_documents(list(df['text'].values))
    db = FAISS.from_documents(docs, embeddings)
    context = db.similarity_search(question, k=4)
    return context


@st.cache(allow_output_mutation=True)
def start_app():
    with st.spinner("Loading model. Please hold..."):
        # context = create_context()
        pipeline = ''
    return pipeline



pdf_files = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    with st.spinner("processing pdf..."):
        df = extract_text_from_pdfs(pdf_files)
        # context = create_context(df)
        # del df
    # topic = st.text_input("Enter the topic you want to ask here")
    question = st.text_input("Enter your questions here...")

    if question != "":
        # pipeline = get_pipeline()
        with st.spinner("Searching. Please hold..."):
            context = create_context(df)
            qa_pipeline = start_app()
            answer = answer_question(question, context)
            st.write(answer)
        del qa_pipeline

