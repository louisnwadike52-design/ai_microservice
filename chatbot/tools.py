# chatbot/tools.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, root_validator
from typing import Type, Any, Union, Optional, List, Dict
from langchain_core.callbacks import CallbackManagerForToolRun
import logging
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Import the actual query function and collection names from services
from .services import query_vector_store, USER_TRANSACTIONS_COLLECTION, USER_CHAT_HISTORY_COLLECTION

logger = logging.getLogger(__name__)


class VectorStoreQueryInput(BaseModel):
    query: str = Field(description="The query to search for in the vector store.")
    user_id: str = Field(description="The user ID to filter results for.")
    collection_name: str = Field(description="The name of the collection to search in.")
    n_results: int = Field(default=3, description="Number of results to return.")


def get_vector_store(collection_name: str) -> FAISS:
    """Initialize and return the vector store for a specific collection.
    
    Args:
        collection_name: Name of the collection to initialize or load
        
    Returns:
        FAISS: Initialized vector store
        
    This function will:
    1. Try to load an existing index
    2. Create a new one if none exists
    3. Handle edge cases like empty collections
    4. Ensure the vector store is always in a valid state
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Force CPU to avoid MPS issues
    )
    
    # Create vector store directory if it doesn't exist
    vector_store_dir = f"vector_store/{collection_name}"
    os.makedirs(vector_store_dir, exist_ok=True)
    
    try:
        # Try to load existing index
        if os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
            try:
                vector_store = FAISS.load_local(
                    folder_path=vector_store_dir,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True  # Safe because we control the directory
                )
                logger.info(f"Successfully loaded existing FAISS index from {vector_store_dir}")
                return vector_store
            except Exception as e:
                logger.warning(f"Error loading existing FAISS index: {e}. Creating new one.")
                # If loading fails, we'll create a new index below
        
        # Create new index with system metadata
        current_time = datetime.now().isoformat()
        metadata = {
            "type": "system",
            "user_id": "system",
            "created_at": current_time,
            "collection": collection_name
        }
        
        # Initialize with a system document
        initial_doc = f"System document for {collection_name} collection. Created at {current_time}"
        
        vector_store = FAISS.from_texts(
            texts=[initial_doc],
            embedding=embeddings,
            metadatas=[metadata]
        )
        
        # Save the new index
        vector_store.save_local(vector_store_dir)
        logger.info(f"Created and saved new FAISS index in {vector_store_dir}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Critical error in get_vector_store: {e}")
        # Create a minimal working index as fallback
        try:
            vector_store = FAISS.from_texts(
                texts=["Emergency fallback document"],
                embedding=embeddings,
                metadatas=[{"type": "system", "user_id": "system", "status": "fallback"}]
            )
            logger.warning("Created emergency fallback vector store")
            return vector_store
        except Exception as fallback_error:
            logger.critical(f"Failed to create fallback vector store: {fallback_error}")
            raise RuntimeError("Could not initialize vector store in any way") from fallback_error


class VectorStoreQueryTool(BaseTool):
    """Tool to query the user's transaction documents and chat history stored in separate vector database collections."""
    name: str = "query_user_context"
    description: str = "Search the user's specific financial documents (transactions, etc.) AND their recent chat history. This tool provides context from both sources (history first, then transactions). Use it for questions about personal data, balances, spending, past transactions, or things mentioned in your previous chat history."
    
    # Define the fields that can be set during initialization
    user_id: str = Field(description="The user ID to filter results for.")
    tx_history: List[Dict[str, Any]] = Field(default_factory=list, description="The user's transaction history.")
    
    def __init__(self, user_id: str, tx_history: List[Dict[str, Any]]):
        """Initialize the tool with user ID and transaction history."""
        super().__init__()
        self.user_id = user_id
        self.tx_history = tx_history

    def save_to_chat_history(self, user_input: str, assistant_response: str) -> None:
        """Save a chat exchange to the chat history vector store."""
        try:
            # Create documents for both user input and assistant response
            user_doc = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            
            assistant_doc = {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metadata
            metadata = {
                "user_id": self.user_id,
                "type": "chat",
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to chat history vector store
            from .services import save_to_vector_store
            save_to_vector_store(
                collection_name=USER_CHAT_HISTORY_COLLECTION,
                texts=[json.dumps(user_doc), json.dumps(assistant_doc)],
                metadatas=[metadata, metadata]
            )
            
            logger.info(f"Saved chat exchange to history for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving chat history for user {self.user_id}: {e}")
            raise

    def save_to_transaction_history(self, transaction: Dict[str, Any]) -> None:
        """Save a transaction to the transaction history vector store."""
        try:
            # Add metadata
            metadata = {
                "user_id": self.user_id,
                "type": "transaction",
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to transaction history vector store
            from .services import save_to_vector_store
            save_to_vector_store(
                collection_name=USER_TRANSACTIONS_COLLECTION,
                texts=[json.dumps(transaction)],
                metadatas=[metadata]
            )
            
            logger.info(f"Saved transaction to history for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving transaction history for user {self.user_id}: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history from the vector store."""
        try:
            results = query_vector_store(
                collection_name=USER_CHAT_HISTORY_COLLECTION,
                query="chat history",
                user_id=self.user_id,
                doc_type="chat"
            )
            
            # Parse the JSON strings back into dictionaries
            parsed_results = []
            for result in results:
                try:
                    # If result is already a dict, use it directly
                    if isinstance(result, dict):
                        parsed_results.append(result)
                        continue
                        
                    # If result is a string, try to parse it
                    if isinstance(result, str):
                        parsed = json.loads(result)
                        parsed_results.append(parsed)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing chat history result: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in chat history parsing: {e}")
                    continue
            
            return parsed_results
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """Get transaction history from the vector store."""
        try:
            results = query_vector_store(
                collection_name=USER_TRANSACTIONS_COLLECTION,
                query="transaction history",
                user_id=self.user_id,
                doc_type="transaction"
            )
            
            # Parse the JSON strings back into dictionaries
            parsed_results = []
            for result in results:
                try:
                    parsed = json.loads(result)
                    parsed_results.append(parsed)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing transaction history result: {e}")
                    continue
            
            return parsed_results
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool to search both chat and transaction history."""
        try:
            # Get chat history first
            chat_history = self.get_chat_history()
            
            # Get transaction history from vector store
            stored_tx_history = self.get_transaction_history()
            
            # Combine results
            results = []
            
            # Add chat history first if exists
            if chat_history:
                for entry in chat_history:
                    if isinstance(entry, dict) and "role" in entry and "content" in entry:
                        results.append(f"{entry['role']}: {entry['content']}")
            
            # Add stored transaction history if exists
            if stored_tx_history:
                for tx in stored_tx_history:
                    if isinstance(tx, dict):
                        results.append(f"Stored Transaction: {json.dumps(tx)}")
            
            # Always add tx_history from request payload
            if self.tx_history:
                for tx in self.tx_history:
                    if isinstance(tx, dict):
                        results.append(f"Recent Transaction: {json.dumps(tx)}")
            
            return "\n".join(results) if results else "No relevant history found."
            
        except Exception as e:
            logger.error(f"Error in VectorStoreQueryTool: {e}")
            return f"Error searching history: {str(e)}"

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run the tool asynchronously."""
        raise NotImplementedError("VectorStoreQueryTool does not support async")

    def run(self, query: str, **kwargs) -> str:
        """Run the tool to search both chat and transaction history."""
        return self._run(query)

    # If you need async execution, define _arun as well
    # async def _arun(
    #     self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: Any
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     # Implement async version of query_vector_store if needed
    #     raise NotImplementedError("VectorStoreQueryTool does not support async") 
