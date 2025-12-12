from ninja import NinjaAPI, Schema
from ninja.errors import ValidationError
from pydantic import Field, field_validator  # Import Field for validation/defaults if needed later
from . import services  # Import the services module
# from .rag_pipeline import rag_chain_instance # No longer needed
from .agent import (
    get_agent_response, cleanup_old_memories, export_chat_history,
    import_chat_history, compress_memory, merge_chat_histories,
    search_chat_history, backup_all_memories, restore_from_backup,
    list_available_backups, schedule_automatic_backup, get_backup_schedule
)
import logging
from ninja.errors import HttpError # Import HttpError at the top
from typing import List, Dict, Any, Optional, Literal
import json
from datetime import datetime

# Import collection names from services
from .services import USER_TRANSACTIONS_COLLECTION, USER_CHAT_HISTORY_COLLECTION

# Import AI Scan router
from .ai_scan_api import ai_scan_router

api = NinjaAPI()

logger = logging.getLogger(__name__)

# Add global error handler for validation errors
@api.exception_handler(ValidationError)
def validation_error_handler(request, exc):
    logger.error(f"Validation error for {request.method} {request.path}: {exc}")
    logger.error(f"Request body: {request.body.decode('utf-8') if hasattr(request.body, 'decode') else str(request.body)}")
    return api.create_response(
        request,
        {"detail": f"Validation error: {str(exc)}", "errors": exc.errors if hasattr(exc, 'errors') else str(exc)},
        status=422,
    )

# --- Schemas ---
class TransactionFilePathSchema(Schema):
    file_path: str
    user_id: str

class ChatHistoryFilePathSchema(Schema):
    file_path: str
    user_id: str

class AddTransactionSchema(Schema):
    user_id: str
    transaction: Dict[str, Any] = Field(description="Transaction data to add to user's transaction history")

class QuerySchema(Schema):
    query: str
    user_id: str
    access_token: Optional[str] = Field(default=None, description="Access token for external API calls")
    tx_history: List[Dict[str, Any]] = Field(default=[], description="Recent transactions for context (last 5 transactions)")

    @field_validator('tx_history', mode='before')
    @classmethod
    def parse_tx_history(cls, v):
        """Parse tx_history if it's sent as a JSON string."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, list):
                    raise ValueError("tx_history must be a list")
                return parsed
            except json.JSONDecodeError:
                raise ValueError("tx_history must be a valid JSON list")
        return v

class ResponseSchema(Schema):
    response: str

class MemoryManagementSchema(Schema):
    user_id: str
    days_threshold: Optional[int] = Field(default=30, description="Number of days of inactivity before cleanup")

class ChatHistoryExportSchema(Schema):
    user_id: str
    format: str = Field(default="json", description="Export format (json or txt)")

class ChatHistoryImportSchema(Schema):
    user_id: str
    history: List[Dict[str, str]] = Field(description="Chat history to import")

class MergeHistoriesSchema(Schema):
    user_id: str
    histories: List[List[Dict[str, str]]] = Field(description="List of chat histories to merge")
    deduplicate: bool = Field(default=True, description="Whether to remove duplicate messages")

class SearchHistorySchema(Schema):
    user_id: str
    query: str
    max_results: int = Field(default=10, description="Maximum number of results to return")

class BackupResponseSchema(Schema):
    backup_path: str
    message: str

class BackupScheduleSchema(Schema):
    time: str = Field(description="Time of day for backup (24-hour format, e.g., '14:30')")
    days: List[int] = Field(default=[0,1,2,3,4,5,6], description="Days of week to run backup (0=Monday, 6=Sunday)")

class BackupInfoSchema(Schema):
    filename: str
    path: str
    size_bytes: int
    created_at: str
    modified_at: str

class EnhancedSearchSchema(Schema):
    user_id: str
    query: str
    max_results: int = Field(default=10, description="Maximum number of results to return")
    start_date: Optional[str] = Field(default=None, description="Start date for search (ISO format)")
    end_date: Optional[str] = Field(default=None, description="End date for search (ISO format)")
    message_type: Optional[Literal["user", "assistant", "all"]] = Field(
        default="all",
        description="Type of messages to search (user, assistant, or all)"
    )

# --- Endpoints will go below ---

@api.post("/index_transactions")
def index_transactions(request, data: TransactionFilePathSchema):
    """Indexes a transaction file for a given user into the transactions collection."""
    try:
        success = services.index_document(
            file_path=data.file_path,
            user_id=data.user_id,
            # Pass the specific collection name, remove metadata argument
            collection_name=USER_TRANSACTIONS_COLLECTION 
        )
        if success:
            return {"message": f"Successfully indexed transactions file {data.file_path} for user {data.user_id}"}
        else:
            logger.error(f"services.index_document returned False for transactions file {data.file_path}, user {data.user_id}")
            raise HttpError(500, f"Failed to index transactions file {data.file_path} for user {data.user_id}. Check server logs.")
    except Exception as e:
        logger.error(f"Error indexing transactions file {data.file_path} for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Internal server error indexing transactions file. Details: {e}")

@api.post("/index_chat_history")
def index_chat_history(request, data: ChatHistoryFilePathSchema):
    """Indexes a chat history file for a given user into the chat history collection."""
    try:
        success = services.index_document(
            file_path=data.file_path,
            user_id=data.user_id,
            # Pass the specific collection name, remove metadata argument
            collection_name=USER_CHAT_HISTORY_COLLECTION
        )
        if success:
            return {"message": f"Successfully indexed chat history file {data.file_path} for user {data.user_id}"}
        else:
            logger.error(f"services.index_document returned False for chat history file {data.file_path}, user {data.user_id}")
            raise HttpError(500, f"Failed to index chat history file {data.file_path} for user {data.user_id}. Check server logs.")
    except Exception as e:
        logger.error(f"Error indexing chat history file {data.file_path} for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Internal server error indexing chat history file. Details: {e}")

@api.post("/chat", response=ResponseSchema)
def chat(request, data: QuerySchema = None):
    """Main chat endpoint that handles user queries."""
    try:
        # Handle both JSON and multipart form data
        if data is None:
            # Manually parse request for multipart/form-data
            query = request.POST.get('query', '')
            user_id = request.POST.get('user_id', '')
            access_token = request.POST.get('access_token', None)
            tx_history_str = request.POST.get('tx_history', '[]')

            # Parse tx_history from JSON string
            try:
                tx_history = json.loads(tx_history_str) if isinstance(tx_history_str, str) else tx_history_str
                if not isinstance(tx_history, list):
                    tx_history = []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tx_history, using empty list: {tx_history_str}")
                tx_history = []
        else:
            # Use validated schema data
            query = data.query
            user_id = data.user_id
            tx_history = data.tx_history
            access_token = data.access_token

        response_text = get_agent_response(
            user_input=query,
            user_id=user_id,
            tx_history=tx_history,
            access_token=access_token
        )
        logger.info(f"Agent response for user {user_id}: {response_text}")
        return ResponseSchema(response=response_text)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HttpError(500, f"Error processing chat request: {str(e)}")

@api.post("/add-transaction")
def add_transaction(request, data: AddTransactionSchema):
    """Add a new transaction to the user's transaction history vector store."""
    # Log the raw request body for debugging
    logger.info(f"Received add-transaction request - Raw body: {request.body.decode('utf-8') if hasattr(request.body, 'decode') else str(request.body)}")
    
    try:
        # Log the parsed data for debugging
        logger.info(f"Parsed data - user_id: {data.user_id}, transaction keys: {list(data.transaction.keys()) if data.transaction else 'None'}")
        
        # Add timestamp if not present
        transaction_data = data.transaction.copy()
        if "timestamp" not in transaction_data:
            transaction_data["timestamp"] = datetime.now().isoformat()
        
        # Prepare metadata for the vector store
        metadata = {
            "user_id": data.user_id,
            "type": "transaction",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to transaction history vector store
        services.save_to_vector_store(
            collection_name=USER_TRANSACTIONS_COLLECTION,
            texts=[json.dumps(transaction_data)],
            metadatas=[metadata]
        )
        
        logger.info(f"Successfully added transaction to history for user {data.user_id}: {transaction_data}")
        return {"message": f"Successfully added transaction to history for user {data.user_id}"}
        
    except Exception as e:
        logger.error(f"Error adding transaction for user {getattr(data, 'user_id', 'unknown')}: {e}", exc_info=True)
        raise HttpError(500, f"Error adding transaction: {str(e)}")

@api.post("/memory/cleanup")
def cleanup_memory(request, data: MemoryManagementSchema):
    """Manually trigger memory cleanup for a specific user or all users."""
    try:
        if data.user_id:
            # Cleanup specific user's memory
            cleanup_old_memories(user_id=data.user_id, days_threshold=data.days_threshold)
            return {"message": f"Successfully cleaned up memory for user {data.user_id}"}
        else:
            # Cleanup all memories
            cleanup_old_memories(days_threshold=data.days_threshold)
            return {"message": "Successfully cleaned up all memories"}
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}", exc_info=True)
        raise HttpError(500, f"Error during memory cleanup: {str(e)}")

@api.post("/memory/compress")
def compress_user_memory(request, data: MemoryManagementSchema):
    """Compress a user's memory to reduce disk usage."""
    try:
        compress_memory(data.user_id)
        return {"message": f"Successfully compressed memory for user {data.user_id}"}
    except Exception as e:
        logger.error(f"Error compressing memory for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Error compressing memory: {str(e)}")

@api.get("/memory/export")
def export_memory(request, data: ChatHistoryExportSchema):
    """Export a user's chat history."""
    try:
        history = export_chat_history(data.user_id, data.format)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error exporting memory for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Error exporting memory: {str(e)}")

@api.post("/memory/import")
def import_memory(request, data: ChatHistoryImportSchema):
    """Import chat history for a user."""
    try:
        import_chat_history(data.user_id, data.history)
        return {"message": f"Successfully imported chat history for user {data.user_id}"}
    except Exception as e:
        logger.error(f"Error importing memory for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Error importing memory: {str(e)}")

@api.post("/memory/merge")
def merge_memories(request, data: MergeHistoriesSchema):
    """Merge multiple chat histories for a user."""
    try:
        merge_chat_histories(data.user_id, data.histories, data.deduplicate)
        return {"message": f"Successfully merged {len(data.histories)} histories for user {data.user_id}"}
    except Exception as e:
        logger.error(f"Error merging histories for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Error merging histories: {str(e)}")

@api.post("/memory/search")
def search_memory(request, data: EnhancedSearchSchema):
    """Search through a user's chat history with advanced filtering."""
    try:
        results = search_chat_history(
            user_id=data.user_id,
            query=data.query,
            max_results=data.max_results,
            start_date=data.start_date,
            end_date=data.end_date,
            message_type=data.message_type
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching memory for user {data.user_id}: {e}", exc_info=True)
        raise HttpError(500, f"Error searching memory: {str(e)}")

@api.post("/memory/backup", response=BackupResponseSchema)
def backup_memories(request):
    """Create a backup of all memories."""
    try:
        backup_path = backup_all_memories()
        return BackupResponseSchema(
            backup_path=backup_path,
            message="Successfully created backup of all memories"
        )
    except Exception as e:
        logger.error(f"Error creating memory backup: {e}", exc_info=True)
        raise HttpError(500, f"Error creating backup: {str(e)}")

@api.post("/memory/restore")
def restore_memories(request, data: dict):
    """Restore memories from a backup file."""
    try:
        if "backup_path" not in data:
            raise HttpError(400, "backup_path is required")
        restore_from_backup(data["backup_path"])
        return {"message": f"Successfully restored memories from backup {data['backup_path']}"}
    except Exception as e:
        logger.error(f"Error restoring from backup: {e}", exc_info=True)
        raise HttpError(500, f"Error restoring from backup: {str(e)}")

@api.get("/memory/backups", response=List[BackupInfoSchema])
def list_backups(request):
    """List all available backup files."""
    try:
        backups = list_available_backups()
        return backups
    except Exception as e:
        logger.error(f"Error listing backups: {e}", exc_info=True)
        raise HttpError(500, f"Error listing backups: {str(e)}")

@api.post("/memory/schedule-backup")
def schedule_backup(request, data: BackupScheduleSchema):
    """Schedule automatic backups."""
    try:
        schedule_automatic_backup(data.time, data.days)
        return {"message": f"Successfully scheduled backups for {data.time} on days {data.days}"}
    except Exception as e:
        logger.error(f"Error scheduling backup: {e}", exc_info=True)
        raise HttpError(500, f"Error scheduling backup: {str(e)}")

@api.get("/memory/backup-schedule", response=BackupScheduleSchema)
def get_schedule(request):
    """Get the current backup schedule."""
    try:
        schedule = get_backup_schedule()
        return schedule
    except Exception as e:
        logger.error(f"Error getting backup schedule: {e}", exc_info=True)
        raise HttpError(500, f"Error getting backup schedule: {str(e)}")

# ==================== AI Scan to Pay Endpoints ====================
# Include AI Scan router for image processing and data extraction
api.add_router("/scan/", ai_scan_router, tags=["AI Scan"])
