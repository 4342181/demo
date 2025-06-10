import os
import json
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import tiktoken
import numpy as np
import pickle # Added for saving/loading Python objects

def main():
    # Load variables from .env file
    load_dotenv()

    # --- Configuration Constants ---
    MAX_CONTEXT_TOKENS_FOR_LLM = 12000
    MAX_CHUNK_SIZE_TOKENS = 500
    OVERLAP_TOKENS = 50
    MODEL_ENCODING_NAME = "cl100k_base"

    # Define a filename for your knowledge base
    KNOWLEDGE_BASE_FILE = "knowledge_base.pkl" # Using .pkl for pickle file

    # Initialize tiktoken encoder
    try:
        encoding = tiktoken.get_encoding(MODEL_ENCODING_NAME)
    except Exception as e:
        print(f"Error initializing tiktoken encoder: {e}")
        print("Please ensure 'tiktoken' is installed (pip install tiktoken) and MODEL_ENCODING_NAME is valid.")
        return

    # --- Azure Document Intelligence Credentials and Client Initialization ---
    try:
        di_endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
        di_key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
        if not di_endpoint or not di_key:
            raise ValueError("Document Intelligence environment variables not set.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set DOCUMENTINTELLIGENCE_ENDPOINT and DOCUMENTINTELLIGENCE_API_KEY in your .env file.")
        return

    try:
        di_credential = AzureKeyCredential(di_key)
        di_client = DocumentIntelligenceClient(endpoint=di_endpoint, credential=di_credential)
        di_model_id = "prebuilt-layout"
        print(f"Document Intelligence client initialized with model: {di_model_id}")
    except Exception as e:
        print(f"Error initializing Azure Document Intelligence client: {e}")
        return

    # --- Azure Blob Storage Credentials and Client Initialization ---
    try:
        blob_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

        if not blob_storage_connection_string or not blob_container_name:
            raise ValueError("Azure Blob Storage environment variables not set.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set AZURE_STORAGE_CONNECTION_STRING and AZURE_BLOB_CONTAINER_NAME in your .env file.")
        return

    try:
        blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
        blob_container_client = blob_service_client.get_container_client(blob_container_name)
        print(f"Azure Blob Storage client initialized for container: {blob_container_name}")
    except Exception as e:
        print(f"Error initializing Azure Blob Storage client: {e}")
        return

    # --- Azure OpenAI Service Credentials and Client Initialization ---
    try:
        openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_chat_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        openai_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        openai_api_version = "2024-02-01"

        if not openai_endpoint or not openai_key or not openai_chat_deployment_name or not openai_embedding_deployment_name:
            raise ValueError("One or more Azure OpenAI environment variables not set or empty.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME in your .env file.")
        return

    try:
        openai_client = AzureOpenAI(
            api_version=openai_api_version,
            azure_endpoint=openai_endpoint,
            api_key=openai_key
        )
        print(f"Azure OpenAI chat client initialized with deployment: {openai_chat_deployment_name}")
        print(f"Azure OpenAI embedding client initialized with deployment: {openai_embedding_deployment_name}")
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {e}")
        return

    # --- Embedding Function ---
    def get_embedding(text, model_name=openai_embedding_deployment_name):
        """Generates an embedding for the given text using Azure OpenAI."""
        try:
            response = openai_client.embeddings.create(input=text, model=model_name)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for text (first 50 chars: '{text[:50]}...'): {e}")
            return None

    # --- Text Chunking Function ---
    def chunk_text(text, source_doc_name, max_tokens=MAX_CHUNK_SIZE_TOKENS, overlap=OVERLAP_TOKENS):
        """
        Splits text into chunks of specified token size with overlap.
        Returns a list of dictionaries, each containing 'text_content' and 'source_doc_name'.
        """
        tokens = encoding.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append({"text_content": chunk_text, "source_doc_name": source_doc_name})
            if i + max_tokens >= len(tokens):
                break
            i += (max_tokens - overlap)
            if i < 0: i = 0
        return chunks

    # --- Helper functions for data extraction (from previous version) ---
    def extract_general_document_data(analyze_result):
        extracted_data = {
            "content": analyze_result.content if analyze_result.content else "",
            "paragraphs": [],
            "tables": [],
            "key_value_pairs": [],
        }
        if analyze_result.paragraphs:
            for paragraph in analyze_result.paragraphs:
                if paragraph.content:
                    extracted_data["paragraphs"].append(paragraph.content)
        if analyze_result.key_value_pairs:
            for kvp in analyze_result.key_value_pairs:
                if kvp.key and kvp.value:
                    extracted_data["key_value_pairs"].append({
                        "key": kvp.key.content,
                        "value": kvp.value.content
                    })
        if analyze_result.tables:
            for table_idx, table in enumerate(analyze_result.tables):
                table_str_rows = []
                if table.cells:
                    max_cols = max(cell.column_index for cell in table.cells) + 1 if table.cells else 0
                    grid = [['' for _ in range(max_cols)] for _ in range(table.row_count)]
                    for cell in table.cells:
                        grid[cell.row_index][cell.column_index] = cell.content or ""
                    for r_idx, row in enumerate(grid):
                        table_str_rows.append(" | ".join(row))
                        if r_idx == 0 and table.row_count > 1:
                            table_str_rows.append("---" * max_cols)
                extracted_data["tables"].append(f"Table {table_idx + 1}:\n" + "\n".join(table_str_rows))
        return extracted_data

    # --- Phase 1: Document Processing and Knowledge Base Creation (Indexing) ---
    knowledge_base = []

    # Check if the knowledge base file already exists
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"\n--- Loading Knowledge Base from '{KNOWLEDGE_BASE_FILE}' ---")
        try:
            with open(KNOWLEDGE_BASE_FILE, "rb") as f:
                knowledge_base = pickle.load(f)
            print(f"Knowledge base loaded successfully with {len(knowledge_base)} chunks.")
            # Ensure embeddings are numpy arrays after loading (pickle usually handles this, but good practice)
            for item in knowledge_base:
                if not isinstance(item["embedding"], np.ndarray):
                    item["embedding"] = np.array(item["embedding"])
        except Exception as e:
            print(f"Error loading knowledge base: {e}. Proceeding to rebuild knowledge base.")
            knowledge_base = [] # Reset if loading fails, forcing a rebuild
    else:
        print("\n--- Phase 1: Building Knowledge Base from Documents ---")
        blobs_to_analyze = []
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.docx', '.xlsx', '.pptx', '.html']

        try:
            for blob in blob_container_client.list_blobs():
                blob_name = blob.name
                if any(blob_name.lower().endswith(ext) for ext in supported_extensions):
                    blobs_to_analyze.append(blob_name)
                    print(f"Found supported document: {blob_name}")
                else:
                    print(f"Skipping unsupported file type: {blob_name}")

            if not blobs_to_analyze:
                print(f"No supported documents found in container '{blob_container_name}'. Exiting.")
                return

        except Exception as e:
            print(f"Error listing blobs in container: {e}")
            return

        for document_blob_name in blobs_to_analyze:
            print(f"\n{'='*60}\nProcessing for Knowledge Base: {document_blob_name}\n{'='*60}")
            try:
                print(f"Attempting to download '{document_blob_name}'...")
                blob_client = blob_container_client.get_blob_client(document_blob_name)
                download_stream = blob_client.download_blob()
                document_content_bytes = download_stream.readall()
                print(f"'{document_blob_name}' downloaded. Size: {len(document_content_bytes)} bytes.")

                print(f"Initiating Document Intelligence analysis for '{document_blob_name}'...")
                poller = di_client.begin_analyze_document(model_id=di_model_id, body=document_content_bytes)
                di_result = poller.result()
                print("Document Intelligence analysis complete.")

                extracted_content_structured = extract_general_document_data(di_result)
                full_document_text = extracted_content_structured["content"]

                if extracted_content_structured["key_value_pairs"]:
                    full_document_text += "\n\n--- Key-Value Pairs ---\n"
                    for kvp in extracted_content_structured["key_value_pairs"]:
                        full_document_text += f"{kvp['key']}: {kvp['value']}\n"
                if extracted_content_structured["tables"]:
                    full_document_text += "\n--- Tables ---\n"
                    for table_str in extracted_content_structured["tables"]:
                        full_document_text += table_str + "\n\n"

                if not full_document_text.strip():
                    print(f"WARNING: Document Intelligence extracted very little or no meaningful content from '{document_blob_name}'. Skipping for knowledge base.")
                    continue

                print(f"Chunking document '{document_blob_name}' into smaller pieces...")
                chunks = chunk_text(full_document_text, document_blob_name)
                print(f"Generated {len(chunks)} chunks from '{document_blob_name}'.")

                # Generate embeddings for each chunk and add to knowledge base
                for i, chunk in enumerate(chunks):
                    print(f"Generating embedding for chunk {i+1}/{len(chunks)} of '{document_blob_name}'...")
                    embedding = get_embedding(chunk["text_content"])
                    if embedding:
                        knowledge_base.append({
                            "text": chunk["text_content"],
                            "source": chunk["source_doc_name"],
                            "embedding": np.array(embedding) # Store as numpy array
                        })
                    else:
                        print(f"Skipped chunk {i+1} due to embedding generation error.")

            except Exception as e:
                print(f"ERROR: Failed to process '{document_blob_name}' for knowledge base: {e}")
                continue

        if not knowledge_base:
            print("\nKnowledge base is empty. No documents were processed successfully for RAG. Exiting.")
            return

        print(f"\n--- Knowledge base built with {len(knowledge_base)} chunks from {len(blobs_to_analyze)} documents. ---")

        # Save the knowledge base after building it
        try:
            print(f"Saving knowledge base to '{KNOWLEDGE_BASE_FILE}'...")
            with open(KNOWLEDGE_BASE_FILE, "wb") as f: # "wb" for write binary mode
                pickle.dump(knowledge_base, f)
            print("Knowledge base saved successfully.")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")


    # --- Phase 2: Interactive Chat with Knowledge Base (Retrieval Augmented Generation) ---
    print("\n--- Starting interactive chat with the Knowledge Base ---")
    print("Type your questions and press Enter. Type 'quit' to end the chat.")

    # System message sets the AI's role and instructions.
    system_message = (
    "You are a specialized AI assistant for the Think Tank Sales Team. Your role is to assist with RFP and RFQ analysis and proposal preparation. You should:\n\n"
    "Ingest and summarize RFP/RFQ documents provided by clients.\n"
    "Extract key insights such as:\n"
    "- Technology requirements (platforms, tools, integrations)\n"
    "- User and scale details (number of users, locations, volumes)\n"
    "- Pain points and problems to be solved\n"
    "- Current systems, workflows, or challenges\n"
    "- Actionable tasks for the proposal response team\n"
    "Recommend a draft Bill of Materials (BOM) by comparing with historical quotes and proposals stored in Think Tank’s SharePoint or MS Teams.\n"
    "Output structured summaries and checklists that help the sales team respond faster and more accurately.\n"
    "You operate within a secure, Microsoft-integrated environment (Microsoft Copilot & Azure AI). Prioritize clarity, accuracy, and contextual understanding of business documents. Ensure all outputs are professional and ready for internal use or proposal integration."
)

    while True:
        user_query = input("\nYour question for the knowledge base: ")
        if user_query.lower() == "quit":
            print("Ending chat. Goodbye!")
            break

        try:
            # 1. Embed the user's query
            query_embedding = get_embedding(user_query)
            if query_embedding is None:
                print("Could not generate embedding for your query. Please try again.")
                continue
            query_embedding = np.array(query_embedding)

            # 2. Retrieve relevant chunks from the knowledge base
            # Calculate cosine similarity for all chunks
            similarities = []
            for item in knowledge_base:
                # Ensure embedding is not None and is a numpy array
                if item["embedding"] is not None:
                    # Calculate dot product (cosine similarity for normalized vectors)
                    similarity = np.dot(query_embedding, item["embedding"])
                    similarities.append((similarity, item))

            # Sort by similarity in descending order
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Build context from top relevant chunks until MAX_CONTEXT_TOKENS_FOR_LLM is reached
            retrieved_context = []
            current_context_tokens = 0
            sources = set() # To keep track of unique source documents

            print(f"\nRetrieving relevant information...")
            for sim, item in similarities:
                chunk_tokens = encoding.encode(item["text"])
                if current_context_tokens + len(chunk_tokens) <= MAX_CONTEXT_TOKENS_FOR_LLM:
                    retrieved_context.append(item["text"])
                    current_context_tokens += len(chunk_tokens)
                    sources.add(item["source"])
                else:
                    # If even the first chunk exceeds the limit, truncate it to fit.
                    # This case should ideally be rare if MAX_CHUNK_SIZE_TOKENS is well-tuned.
                    if not retrieved_context and len(chunk_tokens) > MAX_CONTEXT_TOKENS_FOR_LLM:
                        print(f"Warning: First relevant chunk ({len(chunk_tokens)} tokens) exceeds max context limit ({MAX_CONTEXT_TOKENS_FOR_LLM}). Truncating the first chunk.")
                        truncated_chunk_tokens = encoding.encode(item["text"])[:MAX_CONTEXT_TOKENS_FOR_LLM]
                        retrieved_context.append(encoding.decode(truncated_chunk_tokens))
                        current_context_tokens += len(truncated_chunk_tokens)
                        sources.add(item["source"])
                    break # Stop adding chunks if the next one exceeds the limit

            if not retrieved_context:
                print("No relevant information found in the knowledge base that fits the context window.")
                # Fallback to general AI answer if no context is retrieved
                messages_to_send_for_fallback = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}
                ]
                print("AI (no specific document context):")
                response = openai_client.chat.completions.create(
                    messages=messages_to_send_for_fallback,
                    model=openai_chat_deployment_name,
                    temperature=0.7,
                    max_tokens=500
                )
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    print(response.choices[0].message.content)
                else:
                    print("Could not generate a fallback response.")
                continue


            # 3. Augment the prompt with retrieved context
            context_string = "\n\n".join(retrieved_context)
            if sources:
                context_string += f"\n\n--- Source Documents: {', '.join(sources)} ---"

            augmented_user_message = (
                f"Based on the following context, answer the question. "
                f"If the information is not explicitly in the context, say 'I cannot find that information in the provided documents.'\n\n"
                f"Context:\n{context_string}\n\n"
                f"Question: {user_query}"
            )

            messages_to_send_to_llm = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": augmented_user_message}
            ]

            print(f"Sending {current_context_tokens + len(encoding.encode(user_query)) + len(encoding.encode(system_message))} tokens to LLM (estimated for this turn).")
            print(f"Sources used: {', '.join(sources) if sources else 'None'}")

            # 4. Generate the response
            response = openai_client.chat.completions.create(
                messages=messages_to_send_to_llm,
                model=openai_chat_deployment_name,
                temperature=0.0, # Lower temperature for factual retrieval
                max_tokens=1000
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                assistant_response = response.choices[0].message.content
                print(f"AI: {assistant_response}")
            else:
                print("AI: No content received from Azure OpenAI or response was empty.")

        except Exception as e:
            print(f"An error occurred during Azure OpenAI chat completion for query: {e}")

if __name__ == "__main__":
    main()