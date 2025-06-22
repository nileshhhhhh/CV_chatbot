from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import chainlit as cl
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types


DB_FAISS_PATH = 'vector_store'

custom_prompt_template = """You are a helpful assistant tasked with answering questions about individual candidates based only on the provided context.

Each chunk of context includes the name of the candidate it belongs to. You must ensure your answer is consistent with only one candidate and do not mix information from different people.

If the answer is not in the context, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 4}),
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.0
        )
    )
    return llm

#QA Model Function
def qa_bot(candidate_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_path = os.path.join("vector_store", candidate_name)

    if not os.path.exists(vector_path):
        raise ValueError(f"No vector store found for candidate: {candidate_name}")

    db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Welcome to the CV Chatbot!\nPlease enter the candidate's name to load their profile (e.g., `Nilesh Fonseka`).")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()
    chain = cl.user_session.get("chain")

    # Check for switch command
    if user_input.lower().startswith("switch to "):
        new_candidate = user_input[10:].strip()
        try:
            new_chain = qa_bot(new_candidate)
            cl.user_session.set("chain", new_chain)
            cl.user_session.set("current_candidate", new_candidate)
            await cl.Message(content=f"🔁 Switched to candidate `{new_candidate}`. You can now ask questions.").send()
        except Exception as e:
            await cl.Message(content=f"❌ Failed to switch: {str(e)}").send()
        return

    # If no chain yet, treat this as initial candidate load
    if chain is None:
        candidate_name = user_input.lower()
        try:
            chain = qa_bot(candidate_name)
            cl.user_session.set("chain", chain)
            cl.user_session.set("current_candidate", candidate_name)
            await cl.Message(content=f"✅ Loaded CV for `{candidate_name}`. You can now ask your questions.").send()
        except Exception as e:
            await cl.Message(content=f"❌ Error: {str(e)}").send()
        return

    # If chain is set, treat message as a question
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(user_input, callbacks=[cb])
    answer = res.get("result", "")
    if "Sources:" in answer:
        answer = answer.split("Sources:")[0].strip()
    await cl.Message(content=answer).send()
