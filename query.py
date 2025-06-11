from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # or any other embedding model you used
import os

# Step 1: Define embedding model (should match what you used during saving)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # update if you used another

# Step 2: Load the FAISS vector store
vectorstore = FAISS.load_local("vector_store", embeddings=embedding_model, index_name="index", allow_dangerous_deserialization=True)

# Step 3: Run the query
query = "Show me email of Nilesh Fonseka"
docs = vectorstore.similarity_search(query, k=3)

# Step 4: Print the results
for i, doc in enumerate(docs):
    print(f"\n--- Match {i+1} ---\n{doc.page_content}")
