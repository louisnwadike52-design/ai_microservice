# Standard library imports
import os
import logging
import functools
from typing import Union, Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime

# Third-party imports
import requests  # HTTP requests for URL fetching
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

logger = logging.getLogger(__name__)

# Configuration
FAISS_INDEX_PATH = "faiss_index/"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Using a common Sentence Transformer model
# Define separate collection names
USER_TRANSACTIONS_COLLECTION = "user_transaction_history"
USER_CHAT_HISTORY_COLLECTION = "user_chat_history"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={'device': 'cpu'}  # Force CPU to avoid MPS issues
)

def get_vector_store(collection_name: str) -> FAISS:
    """Initialize and return the vector store for a specific collection."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store directory if it doesn't exist
    vector_store_dir = f"vector_store/{collection_name}"
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Try to load existing index
    try:
        # Check if the index files exist
        index_files = ["index.faiss", "index.pkl"]
        if all(os.path.exists(os.path.join(vector_store_dir, f)) for f in index_files):
            logger.info(f"Loading existing FAISS index for collection: {collection_name}")
            return FAISS.load_local(
                folder_path=vector_store_dir,
                embeddings=embeddings,
                allow_dangerous_deserialization=True  # Safe because we control the directory
            )
        else:
            logger.info(f"Index files not found for collection: {collection_name}. Creating new vector store.")
            raise FileNotFoundError("Index files not found")
    except Exception as e:
        logger.info(f"Creating new FAISS index for collection '{collection_name}': {e}")
        # Create new index if none exists or if there was an error
        new_vector_store = FAISS.from_texts(
            texts=["System initialization document"],  # Dummy document
            embedding=embeddings,
            metadatas=[{"type": "system", "user_id": "system", "collection": collection_name}]
        )
        
        # Save the new vector store immediately
        try:
            new_vector_store.save_local(vector_store_dir)
            logger.info(f"Successfully created and saved new vector store for collection: {collection_name}")
        except Exception as save_error:
            logger.error(f"Error saving new vector store for {collection_name}: {save_error}")
            # Continue anyway, the vector store can still be used in memory
        
        return new_vector_store

def save_to_vector_store(
    collection_name: str,
    texts: List[str],
    metadatas: List[Dict[str, Any]]
) -> None:
    """Save documents to a vector store collection with user-specific initialization check."""
    try:
        # Extract user_id from metadata for user-specific operations
        user_id = metadatas[0].get("user_id") if metadatas else None
        
        # Get or create the vector store
        vector_store = get_vector_store(collection_name)
        
        # Check if this user has any existing data in the vector store
        if user_id and collection_name == USER_TRANSACTIONS_COLLECTION:
            has_user_data = _check_user_has_transactions(vector_store, user_id)
            if not has_user_data:
                logger.info(f"No existing transaction data found for user {user_id}. Initializing user-specific transaction vector store.")
                # The vector store already exists (from get_vector_store), we just need to add the first transaction
        
        # Add the new texts to the vector store
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        # Save the updated index
        vector_store_dir = f"vector_store/{collection_name}"
        os.makedirs(vector_store_dir, exist_ok=True)  # Ensure directory exists
        vector_store.save_local(vector_store_dir)
        
        if user_id:
            logger.info(f"Successfully saved {len(texts)} documents to {collection_name} for user {user_id}")
        else:
            logger.info(f"Successfully saved {len(texts)} documents to {collection_name}")
            
    except Exception as e:
        user_info = f" for user {metadatas[0].get('user_id')}" if metadatas and metadatas[0].get('user_id') else ""
        logger.error(f"Error saving to vector store {collection_name}{user_info}: {e}")
        raise

def _check_user_has_transactions(vector_store: FAISS, user_id: str) -> bool:
    """Check if a user has any existing transaction data in the vector store."""
    try:
        # Try to search for any documents with this user_id and transaction type
        results = vector_store.similarity_search(
            query="transaction",  # Generic query
            k=1,  # We only need to know if any exist
            filter={"user_id": user_id, "type": "transaction"}
        )
        return len(results) > 0
    except Exception as e:
        logger.warning(f"Could not check existing user data for {user_id}: {e}")
        # If we can't check, assume no data exists (safer to initialize)
        return False

def query_vector_store(
    collection_name: str,
    query: str,
    user_id: str,
    doc_type: str,
    k: int = 100
) -> List[Dict[str, Any]]:
    """Query a vector store collection."""
    try:
        vector_store = get_vector_store(collection_name)
        results = vector_store.similarity_search(
            query=query,
            k=k,
            filter={"user_id": user_id, "type": doc_type}
        )
        
        # Parse and sort the results
        parsed_results = []
        for doc in results:
            try:
                entry = json.loads(doc.page_content)
                parsed_results.append(entry)
            except json.JSONDecodeError:
                continue
        
        # Sort by timestamp
        parsed_results.sort(key=lambda x: x.get("timestamp", ""))
        return parsed_results
        
    except Exception as e:
        logger.error(f"Error querying vector store {collection_name}: {e}")
        return []

def get_content_from_source(source: str) -> Tuple[Optional[str], str]:
    """Reads content from a local file path or downloads it from a URL.
    
    Args:
        source: File path or URL to read from
        
    Returns:
        Tuple of (content, filename) where content is None if retrieval failed
    """
    filename = os.path.basename(source)
    content = None

    if source.startswith("http://") or source.startswith("https://"):
        # It's a URL
        try:
            logger.info(f"Downloading content from URL: {source}")
            
            # Check if it's a Google Cloud Storage URL
            if "storage.googleapis.com" in source:
                # Get credentials from service account file
                try:
                    from google.cloud import storage
                    from google.cloud.exceptions import NotFound
                    from google.oauth2 import service_account
                    
                    # Load service account credentials
                    credentials_path = "lazervault-c4d5f9e91078.json"
                    if not os.path.exists(credentials_path):
                        logger.error(f"GCP credentials file not found at: {credentials_path}")
                        raise FileNotFoundError(f"GCP credentials file not found at: {credentials_path}")
                        
                    credentials = service_account.Credentials.from_service_account_file(credentials_path)
                    
                    # Parse bucket and blob path from URL
                    path = urlparse(source).path.lstrip('/')
                    bucket_name = path.split('/')[0]
                    blob_path = '/'.join(path.split('/')[1:])
                    
                    # Initialize storage client with credentials
                    storage_client = storage.Client(credentials=credentials)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    
                    # Download as string
                    content = blob.download_as_text()
                    logger.info(f"Successfully downloaded content from GCS: {source}")
                except ImportError:
                    logger.error("google-cloud-storage package not installed. Falling back to direct HTTP request.")
                    raise
                except NotFound:
                    logger.error(f"File not found in GCS: {source}")
                    return None, filename
                except Exception as e:
                    logger.error(f"Error accessing GCS: {e}")
                    raise
            else:
                # Regular HTTP request
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                content = response.text
                
            # Try to get a more meaningful filename from URL path
            parsed_path = urlparse(source).path
            if parsed_path:
                filename = os.path.basename(parsed_path)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading content from {source}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching {source}: {e}")
    else:
        # Assume it's a local file path
        if not os.path.exists(source):
            logger.error(f"Local file not found at: {source}")
        else:
            try:
                logger.info(f"Reading content from local file: {source}")
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Successfully read content from {source}")
            except Exception as e:
                logger.error(f"Error reading local file {source}: {e}")
                
    return content, filename

def index_document(file_path: str, user_id: str, collection_name: str) -> bool:
    """Reads a document, splits it, and adds it to the specified FAISS index with user_id metadata."""
    if not collection_name:
        logger.error("Collection name not provided to index_document.")
        return False
        
    content, filename = get_content_from_source(file_path)

    if content is None:
        logger.warning(f"Could not retrieve content for source: {file_path}. Aborting indexing for collection '{collection_name}'.")
        return False

    try:
        # Get or create FAISS index
        vectorstore = get_vector_store(collection_name)
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(content)

        if not chunks:
            logger.warning(f"No text chunks generated from source: {file_path}. Content might be empty or non-text.")
            return False

        # Prepare data for FAISS
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "user_id": user_id,
                "source": filename,
                "chunk_id": f"{user_id}_{filename}_{i}",
                "type": "user_document"
            })

        # Add documents to FAISS
        save_to_vector_store(collection_name, texts, metadatas)
        
        logger.info(f"Successfully indexed {len(chunks)} chunks from source '{filename}' into collection '{collection_name}' for user {user_id}")
        return True

    except Exception as e:
        logger.error(f"Error during chunking or indexing document from {file_path} into collection '{collection_name}' for user {user_id}: {e}", exc_info=True)
        return False

def query_vector_store_general(query: str, user_id: str, collection_name: str, n_results: int = 3) -> List[str]:
    """Queries the specified FAISS index for relevant documents for a specific user."""
    if not collection_name:
        logger.error("Collection name not provided to query_vector_store_general.")
        return []
        
    try:
        # Get FAISS index
        vectorstore = get_vector_store(collection_name)
        
        # Search with metadata filter
        results = vectorstore.similarity_search_with_score(
            query,
            k=n_results,
            filter={"user_id": user_id}
        )
        
        if results:
            retrieved_docs = [doc.page_content for doc, _ in results]
            logger.info(f"Retrieved {len(retrieved_docs)} documents from collection '{collection_name}' for user '{user_id}'")
            return retrieved_docs
        else:
            logger.info(f"No documents found in collection '{collection_name}' for user '{user_id}'")
            return [f"I searched through your {collection_name} but couldn't find any information matching your query."]
            
    except Exception as e:
        logger.error(f"Error querying collection '{collection_name}' for user {user_id}: {e}", exc_info=True)
        return [f"I searched through your {collection_name} but couldn't find any information matching your query. This might be because you don't have any {collection_name} yet."]
