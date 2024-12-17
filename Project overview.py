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

# Extract text from ALL pages
def extract_all_pdf_content(pdf_path):
    all_text = []
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        # Extract text from every page
        for page_num in range(len(pdf.pages)):
            page_text = pdf.pages[page_num].extract_text()
            if page_text:
                all_text.append(page_text)

            # Extract tables
            page_table = pdf.pages[page_num].extract_table()
            if page_table:
                all_tables.append(page_table)

    return all_text, all_tables

# Initialize the Sentence-BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract all content from the PDF
all_text, all_tables = extract_all_pdf_content(pdf_path)

# Ensure that there is content in the extracted text and tables
if all_text and all_tables:
    chunks = [all_text[0]] + [str(row) for row in all_tables[0]]  # Modify as needed
else:
    print("No text or tables available to process.")
    chunks = []

# Generate embeddings for the chunks if they exist
if chunks:
    embeddings = model.encode(chunks)
    # Store the embeddings in a FAISS index for similarity-based retrieval
    dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Create an L2 similarity index
    index.add(np.array(embeddings))  # Add embeddings to the FAISS index
    print("Embeddings successfully stored in FAISS index.")
else:
    print("No chunks to process.")

# Function to search for the most relevant chunks based on the user's query
def search_relevant_chunks(query):
    if chunks:
        query_embedding = model.encode([query])

        # Search for the top 3 most similar chunks
        distances, indices = index.search(np.array(query_embedding), k=3)

        # Retrieve the corresponding chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    else:
        return []

# User input for the query
user_query = input("Please enter your query: ")

# Get the relevant chunks based on the user's query
retrieved_chunks = search_relevant_chunks(user_query)

# Display the retrieved chunks if any
if retrieved_chunks:
    print("Relevant Chunks Found:\n", "\n".join(retrieved_chunks))
else:
    print("No relevant chunks found for your query.")

# Load a pre-trained FLAN-T5 model and tokenizer for response generation
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Local LLM model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to generate a response based on the retrieved chunks and the query
def generate_answer_from_retrieved_chunks(retrieved_chunks, query):
    context = "\n".join(retrieved_chunks)
    prompt = f"Based on the following retrieved information:\n{context}\n\nAnswer the following question: {query}"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = llm_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Generate the final answer based on the query and retrieved chunks
if retrieved_chunks:
    final_response = generate_answer_from_retrieved_chunks(retrieved_chunks, user_query)
    print("\nGenerated Answer:\n", final_response)
else:
    print("\nNo answer generated due to lack of relevant chunks.")
