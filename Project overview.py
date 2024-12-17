
import pdfplumber
from google.colab import files
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 3: Upload the PDF file
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Step 4: Extract content from the PDF
with pdfplumber.open(pdf_path) as pdf:
    # Page 2: Extract text
    page2_text = pdf.pages[1].extract_text()
    # Page 6: Extract table data
    page6_table = pdf.pages[5].extract_table()

# Print the extracted text and table for verification
print("Page 2 Text:\n", page2_text)
print("\nPage 6 Table:\n", page6_table)

# Step 5: Initialize the embedding model (Sentence-BERT)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 6: Prepare chunks for embedding
chunks = [page2_text] + [str(row) for row in page6_table]  # Combine text and table rows

# Step 7: Generate embeddings for the chunks
embeddings = model.encode(chunks)

# Step 8: Store embeddings in FAISS index for similarity search
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 similarity index
index.add(np.array(embeddings))  # Add embeddings to FAISS index

print("Embeddings stored in FAISS index.")

# Step 9: Define search function to find the most relevant chunks based on a user query
def search(query, model, index, chunks):
    # Generate query embedding
    query_embedding = model.encode([query])
    
    # Search for the top-3 most similar chunks
    _, indices = index.search(np.array(query_embedding), k=3)
    results = [chunks[i] for i in indices[0]]
    return results

# Step 10: User input for query
user_query = input("Enter your query: ")

# Step 11: Retrieve relevant chunks based on the query
retrieved_chunks = search(user_query, model, index, chunks)

# Print retrieved chunks
print("Retrieved Chunks:\n", retrieved_chunks)

# Step 12: Load the local LLM model and tokenizer (FLAN-T5)
model_name = "google/flan-t5-base"  # Use "google/flan-t5-large" for better results
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Local LLM model loaded successfully!")

# Step 13: Define the function to generate a response using the retrieved chunks and user query
def generate_response_local(results, query):
    # Combine retrieved chunks into a context for the LLM
    context = "\n".join(results)
    prompt = (
        f"Given the following retrieved information:\n{context}\n\n"
        f"Answer the question: {query}"
    )
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a response from the LLM
    outputs = llm.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 14: Generate the final response based on retrieved chunks and user query
final_response = generate_response_local(retrieved_chunks, user_query)

# Print the final generated response
print("\nGenerated Response:\n", final_response)
