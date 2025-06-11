import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Directory containing the extracted text files
txt_folder = "txt_output"
vectorstore_path = "vector_store"

# Step 1: Load text files
documents = []
for file in os.listdir(txt_folder):
    if file.endswith(".txt"):
        with open(os.path.join(txt_folder, file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))

print(f"Total chunks created: {len(chunks)}")

# Step 3: Create Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS vector store
vectorstore = FAISS.from_texts(chunks, embedding_model)
vectorstore.save_local(vectorstore_path)

print(f"Vector store saved to: {vectorstore_path}")