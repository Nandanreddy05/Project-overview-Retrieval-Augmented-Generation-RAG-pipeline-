# Import the necessary libraries
import pdfplumber
from google.colab import files
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Upload the PDF file
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

# Open the PDF and check the number of pages
with pdfplumber.open(pdf_path) as pdf:
    total_pages = len(pdf.pages)
    print(f"Total number of pages in the PDF: {total_pages}")
    
    # Check if there are enough pages
    if total_pages > 1:
        page2_text = pdf.pages[1].extract_text()  # Extract text from page 2
        print("Page 2 Text:\n", page2_text)
    else:
        print("Page 2 does not exist in this PDF.")

    if total_pages > 5:
        page6_table = pdf.pages[5].extract_table()  # Extract table from page 6
        print("\nPage 6 Table:\n", page6_table)
    else:
        print("Page 6 does not exist in this PDF.")

# Initialize the Sentence-BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare the text chunks (combining text from page 2 and table rows from page 6)
chunks = [page2_text] + [str(row) for row in page6_table]

# Generate embeddings for the chunks
embeddings = model.encode(chunks)

# Store the embeddings in a FAISS index for similarity-based retrieval
dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # Create an L2 similarity index
index.add(np.array(embeddings))  # Add embeddings to the FAISS index

print("Embeddings successfully stored in FAISS index.")

# Function to search for the most relevant chunks based on the user's query
def search_relevant_chunks(query):
    # Convert the query into embeddings
    query_embedding = model.encode([query])
    
    # Search for the top 3 most similar chunks
    _, indices = index.search(np.array(query_embedding), k=3)
    
    # Retrieve the corresponding chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# User input for the query
user_query = input("Please enter your query: ")

# Get the relevant chunks based on the user's query
retrieved_chunks = search_relevant_chunks(user_query)

# Display the retrieved chunks
print("Relevant Chunks Found:\n", retrieved_chunks)

# Load a pre-trained FLAN-T5 model and tokenizer for response generation
model_name = "google/flan-t5-base"  # You can use a larger version like 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Local LLM model loaded successfully!")

# Function to generate a response based on the retrieved chunks and the query
def generate_answer_from_retrieved_chunks(retrieved_chunks, query):
    # Combine the relevant chunks into one context string
    context = "\n".join(retrieved_chunks)
    
    # Create the prompt by including the context and the query
    prompt = f"Based on the following retrieved information:\n{context}\n\nAnswer the following question: {query}"

    # Tokenize the prompt for the LLM
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate a response using the model
    output = llm_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    # Decode the response and return it
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Generate the final answer based on the query and retrieved chunks
final_response = generate_answer_from_retrieved_chunks(retrieved_chunks, user_query)

# Display the generated response
print("\nGenerated Answer:\n", final_response)
