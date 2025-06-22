import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Folder paths
txt_folder = "txt_output"
vectorstore_base_path = "vector_store"
os.makedirs(vectorstore_base_path, exist_ok=True)

# Initialize embedding model and splitter
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

# Process each CV separately
for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        candidate_name = os.path.splitext(file)[0]
        file_path = os.path.join(txt_folder, file)

        print(f"Processing {candidate_name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Add candidate name to the beginning of each chunk
        text_with_name = f"Candidate: {candidate_name}\n{raw_text}"
        chunks = splitter.split_text(text_with_name)

        documents = [Document(page_content=chunk, metadata={"source": candidate_name}) for chunk in chunks]

        # Create vector store and save to subdirectory
        candidate_vectorstore_path = os.path.join(vectorstore_base_path, candidate_name)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(candidate_vectorstore_path)

        print(f"✅ Saved vector store for {candidate_name} to: {candidate_vectorstore_path}")
