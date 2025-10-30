# chatbot/agent.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents import AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import logging
from langchain.tools import BaseTool, Tool
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union, Literal, Type
import pickle
from pathlib import Path
import gzip
import shutil
import tarfile
from datetime import datetime
import schedule
import time
import threading
from django.conf import settings
from pydantic import validator
from django.core.cache import cache

# Import VectorStoreQueryTool from tools.py
from .tools import VectorStoreQueryTool
from .services import USER_CHAT_HISTORY_COLLECTION, USER_TRANSACTIONS_COLLECTION

# Load environment variables from .env file
def load_env_vars():
    """Load environment variables and validate required ones."""
    # Force reload of .env file
    load_dotenv(override=True)
    
    # Get and validate GROQ_API_KEY
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your-groq-api-key-here":
        raise ValueError(
            "GROQ_API_KEY is not set or is using the default value. "
            "Please set it in your .env file and run 'source .env' in your terminal."
        )
    return groq_api_key

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
try:
    logger.info("Initializing Groq LLM...")
    llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY, 
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0  # More deterministic responses
    )
    # Test the API key with a simple call
    logger.info("Testing Groq API connection...")
    llm.invoke("test")
    logger.info("Successfully connected to Groq API")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    if "401" in str(e):
        logger.error("Authentication failed. Please verify your GROQ_API_KEY is valid and not expired.")
    raise ValueError(f"Failed to initialize Groq LLM. Please check your GROQ_API_KEY. Error: {e}")

# Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define general tools (can still be global)
search_tool = DuckDuckGoSearchRun()

# Use the BASE_API_URL from Django settings
BASE_API_URL = settings.BASE_API_URL  # This should be the full external API URL

# Banking Tools
class GetSimilarRecipientsInput(BaseModel):
    name: str = Field(description="The first name or full name of the recipient to search for.")

class GetSimilarRecipientsTool(BaseTool):
    name: str = "get_similar_recipients"
    description: str = "Get a list of similar recipients by name to transfer money to. Helps identify the correct recipient if unsure."
    args_schema: Optional[Type[BaseModel]] = GetSimilarRecipientsInput

    def run(self, name: str, **kwargs) -> str:
        """Run the tool to get similar recipients."""
        return self._run(name)

    def _run(self, name: str) -> str:
        logger.info(f"get_similar_recipients - name: {name}")
        url = f"{BASE_API_URL}/recipients/search-by-name" 
        params = {'name': name}
        logger.info(f"get_similar_recipients - url: {url}, params: {params}")
        
        # Get token from environment
        access_token = os.getenv("ACCESS_TOKEN")
        if not access_token:
            logger.error("Access token not found in environment. API call will fail.")
            return "Error: Authentication token is required but not available."
            
        headers = {"Authorization": f"Bearer {access_token}"}
        logger.info("Using access token for get_similar_recipients API call")

        try:
            # Use synchronous requests instead of aiohttp
            import requests
            response = requests.get(url, params=params, headers=headers)
            response_text = response.text
            logger.info(f"Response from get-similar-recipients: {response.status_code} - {response_text}")
            
            if response.status_code == 200:
                try:
                    # Parse the response to check if we have any recipients
                    recipients = json.loads(response_text)
                    if not recipients:
                        return f"Recipient '{name}' not found in your saved recipients."
                    return response_text
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from get_similar_recipients")
                    return "Error: Received invalid data format from the recipients service."
            elif response.status_code == 401:
                logger.error("Authentication failed for get_similar_recipients")
                return "Error: Authentication failed. Please check your access token."
            else:
                # Propagate API error details if possible
                return f"Error: Could not fetch recipients. Status: {response.status_code} - {response_text}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error in get_similar_recipients: {e}")
            return f"Error: Could not connect to the recipients service. {e}"
        except Exception as e:
            logger.error(f"Unexpected error in get_similar_recipients: {e}")
            return f"Error: An unexpected error occurred while fetching recipients. {e}"

class MakeTransferInput(BaseModel):
    amount: str = Field(description="The amount of money to transfer.")
    recipient_id: Optional[str] = Field(default=None, description="The unique ID of the recipient.")
    description: str = Field(default="", description="A short description or memo for the transfer.")
    category: str = Field(default="", description="The category of the transfer (e.g., Shopping, Utilities, Rent).")
    reference: str = Field(default="", description="A reference for the transfer.")
    from_account_id: str = Field(default="1", description="The account ID from which the money is being sent.")
    to_account_id: Optional[str] = Field(default=None, description="The account ID to which the money is being sent (used if recipient_id is not valid or available).")
    scheduled_at: str = Field(default="", description="The scheduled date and time for the transfer in ISO format. Empty string for immediate transfer.")

    @field_validator('recipient_id', 'to_account_id', mode='before')
    @classmethod
    def validate_recipient_or_account(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Ensure either recipient_id or to_account_id is provided."""
        values = info.data
        if not v and not values.get('to_account_id'):
            raise ValueError("Either recipient_id or to_account_id must be provided")
        return v

class MakeTransferTool(BaseTool):
    name: str = "make_transfer"
    description: str = "Transfers money to a specified recipient after gathering all necessary details and user authorization."
    args_schema: Optional[Type[BaseModel]] = MakeTransferInput

    def run(self, **kwargs) -> str:
        """Run the tool to make a transfer."""
        return self._run(**kwargs)

    def _run(self, amount: str, recipient_id: Optional[str] = None, description: str = "", category: str = "", 
             reference: str = "", from_account_id: str = "1", to_account_id: Optional[str] = None, 
             scheduled_at: str = "") -> str:
        logger.info(f"make_transfer called with - amount: {amount}, recipient_id arg: '{recipient_id}', to_account_id arg: '{to_account_id}', from_account_id: {from_account_id}, scheduled_at: {scheduled_at}")

        # Validate that we have either recipient_id or to_account_id
        if not recipient_id and not to_account_id:
            return "Error: Either recipient_id or to_account_id must be provided for the transfer."

        # Validate scheduled_at format if provided
        if scheduled_at and scheduled_at.strip():
            try:
                # Try to parse the ISO format string to validate it
                datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
            except ValueError:
                logger.error(f"Invalid scheduled_at format: {scheduled_at}")
                return "Error: Invalid scheduled date format. Please provide a valid ISO format datetime string."

        # Build base payload with defaults
        payload = {
            "amount": amount,
            "category": category if category else "Miscellaneous",
            "description": description if description else f"Transfer payment to {recipient_id or to_account_id}",
            "from_account_id": from_account_id,
            "reference": reference if reference else "default",
            "scheduled_at": scheduled_at.strip() if scheduled_at else ""
        }

        valid_recipient_id_str = recipient_id.strip() if recipient_id else None
        valid_to_account_id_str = to_account_id.strip() if to_account_id else None

        if valid_recipient_id_str:
            payload["recipient_id"] = valid_recipient_id_str
            logger.info(f"Using recipient_id: {valid_recipient_id_str} for the transfer.")
        elif valid_to_account_id_str:
            payload["to_account_id"] = valid_to_account_id_str
            logger.info(f"Using to_account_id: {valid_to_account_id_str} as recipient_id was not valid.")
        else:
            logger.error("make_transfer: Neither recipient_id nor to_account_id provided a valid identifier.")
            return "Error: A valid recipient identifier (recipient_id or to_account_id) is required for the transfer."

        url = f"{BASE_API_URL}/transfers" 
        logger.info(f"Sending transfer request to {url} with payload: {json.dumps(payload)}")

        # Get token from environment
        access_token = os.getenv("ACCESS_TOKEN")
        if not access_token:
            logger.error("Access token not found in environment. API call will fail.")
            return "Error: Authentication token is required but not available."
            
        headers = {"Authorization": f"Bearer {access_token}"}
        logger.info("Using access token for make_transfer API call")

        try:
            # Use synchronous requests instead of aiohttp
            import requests
            response = requests.post(url, json=payload, headers=headers)
            response_text = response.text
            logger.info(f"Response from make_transfer: {response.status_code} - {response_text}")
            
            if response.status_code in [200, 201]:
                try:
                    json.loads(response_text)
                    return response_text
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON response from make_transfer")
                    return "Error: Received invalid data format from the transfer service."
            elif response.status_code == 401:
                logger.error("Authentication failed for make_transfer")
                return "Error: Authentication failed. Please check your access token."
            else:
                return f"Error: Transfer failed. Status: {response.status_code} - {response_text}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error in make_transfer: {e}")
            return f"Error: Could not connect to the transfer service. {e}"
        except Exception as e:
            logger.error(f"Unexpected error in make_transfer: {e}")
            return f"Error: An unexpected error occurred during the transfer. {e}"

class SignalTransferSuccessInput(BaseModel):
    transaction_response: str = Field(description="The JSON string response received from the successful 'make_transfer' operation.")

class SignalTransferSuccessTool(BaseTool):
    name: str = "signal_transfer_success"
    description: str = "Signals the frontend that a transfer was successful, providing the transaction details. This tool should ONLY be called after a successful 'make_transfer' call."
    args_schema: Optional[Type[BaseModel]] = SignalTransferSuccessInput

    def run(self, transaction_response: str, **kwargs) -> str:
        """Run the tool to signal transfer success."""
        return self._run(transaction_response)

    def _run(self, transaction_response: str) -> str:
        logger.info(f"Tool signal_transfer_success called with: {transaction_response}")
        try:
            # Parse the JSON data
            parsed_data = json.loads(transaction_response)
            
            # In a real implementation, you would send this to your frontend
            # For now, we'll just log it
            logger.info(f"Transfer completed successfully. Transaction details: {json.dumps(parsed_data)}")
            return "Transfer completed successfully and frontend has been notified."

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in signal_transfer_success tool: {e}")
            return f"Error: Invalid transaction data format: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in signal_transfer_success tool: {e}")
            return f"Error: An unexpected error occurred while sending signal: {e}"

# System prompt for the agent - Now includes banking capabilities and transaction history context
system_prompt = """You are a helpful, intelligent fintech AI chatbot. Your purpose is to assist users with questions about their finances, transactions, and past conversations, as well as general knowledge questions that might require up-to-date information.

You have access to the following tools:
1. query_user_context: Use this tool FIRST to search the user's specific financial documents (transactions, etc.) AND their recent chat history. This tool provides context from both sources (history first, then transactions). Use it for questions about personal data, balances, spending, past transactions, or things mentioned in your previous chat history.

2. get_similar_recipients: Use this tool to search for recipients by name when the user wants to transfer money but isn't sure about the exact recipient details.

3. make_transfer: Use this tool to execute money transfers after gathering all necessary details from the user. IMPORTANT: ONLY call this tool if you have a valid recipient_id from a successful recipient search.

4. signal_transfer_success: Use this tool ONLY after a successful make_transfer call to notify the frontend about the completed transfer.

5. duckduckgo_search: ONLY use this tool as a fallback. You should use it IF AND ONLY IF the 'query_user_context' tool was tried first AND the retrieved context was either empty OR clearly irrelevant/insufficient to answer the user's specific question, AND the nature of the user's query suggests it requires current, potentially time-sensitive external information.

## Response Style Guidelines:
1. Always respond in a natural, conversational tone
2. Never show technical details or internal thoughts
3. Keep responses concise and to the point
4. Use simple, clear language
5. Be friendly but professional
6. Avoid technical jargon unless absolutely necessary
7. Never show JSON or technical structures in responses
8. Never mention the tools you're using
9. Never show your thought process
10. Never show action inputs or outputs
11. When a recipient isn't found, simply say "Couldn't find any recipient with name [name]"
12. When asking for more information, be direct and conversational
13. Never show internal tool calls or technical details in responses
14. Keep error messages simple and user-friendly
15. Always respond as if talking to a non-technical user

## Transfer Money Pipeline:
1. **Get Recipient and Amount**: Start by asking 'Who would you like to send money to and how much?' This gets both pieces of essential information in one question.

2. **Find Recipients Tool**: Use the 'get_similar_recipients' tool with the provided name.
   - **If 'get_similar_recipients' returns an error**: Inform the user, 'I had trouble looking up that name. Could you please spell it?'
   - **If no recipients are found**: Tell the user, 'I couldn't find anyone by that name. Could you please check the spelling or try a different name?' and wait for their response. DO NOT proceed with the transfer flow until a valid recipient is found.
   - **If recipients are found**: Proceed to step 3.

3. **Handle Multiple Matches**: If multiple users are found, list them by number (e.g., '1. John Doe, 2. Jane Doe'). Ask 'Which one?' If their choice is ambiguous, ask 'Which John did you mean? John Doe, number 1, or John Smith, number 2?'

4. **Quick Confirmation**: Once a recipient is identified, quickly confirm: 'Sending [Amount] to [Recipient Name]. Is that correct?'

5. **Optional Details**: If confirmed, ask 'Would you like to add any details like description, category, reference, or schedule the transfer? If not, I'll use defaults.'
   - If they say yes, ask for the specific details they want to add
   - If they say no or don't respond, use defaults:
     - Description: 'Transfer payment to [Recipient Name]'
     - Category: 'Miscellaneous'
     - Reference: 'default'
     - Schedule: Immediate transfer (empty string)
   - For scheduled transfers:
     - If user mentions 'transfer now' or similar, use empty string for scheduled_at
     - If user specifies a future date/time, format it as UTC ISO string
     - Validate the datetime format before proceeding

6. **Authorization**: If everything is confirmed, say 'To proceed, please say AUTHORIZE.'

7. **Execute Transfer**: If they say 'AUTHORIZE', call 'make_transfer' with the details.
   - If successful, inform 'Transfer successful!' and call 'signal_transfer_success'
   - If error, say 'Transfer couldn't be completed. Would you like to try again?'

## Critical Rules for Transfer Flow:
1. NEVER call make_transfer without a valid recipient_id from a successful recipient search
2. If no recipients are found, inform the user and ask for a different name
3. If the user wants to try a different name, reset the transfer flow and start over
4. Only proceed to transfer when ALL required information is available and valid
5. The transfer state's can_proceed flag must be True before attempting a transfer
6. If can_proceed is False, ask the user for more information instead of attempting transfer

## Error Handling:
1. If no recipients are found:
   - Inform the user
   - Ask for a different name
   - Reset the transfer flow
   - DO NOT proceed with transfer
   - DO NOT call make_transfer

2. If multiple recipients are found:
   - List them clearly
   - Ask for clarification
   - Wait for user selection
   - DO NOT proceed until a clear selection is made

3. If transfer fails:
   - Explain the error clearly
   - Offer to try again
   - Reset the flow if needed
Give conised answers and be helpful.

Remember: The transfer flow should only proceed when ALL required information is available and valid, and the transfer state's can_proceed flag is True."""

# Constants for memory management
MEMORY_DIR = Path("chat_memories")
MEMORY_EXPIRY_DAYS = 30  # Days after which inactive memory is cleaned up
MAX_MEMORY_SIZE = 1000  # Maximum number of messages to keep in memory
COMPRESSED_MEMORY_DIR = MEMORY_DIR / "compressed"
BACKUP_DIR = MEMORY_DIR / "backups"
BACKUP_SCHEDULE_FILE = MEMORY_DIR / "backup_schedule.json"

# Create memory directories if they don't exist
MEMORY_DIR.mkdir(exist_ok=True)
COMPRESSED_MEMORY_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# Initialize backup scheduler
backup_scheduler = schedule.Scheduler()
scheduler_thread = None

def get_memory_path(user_id: str, compressed: bool = False) -> Path:
    """Get the path for a user's memory file."""
    base_dir = COMPRESSED_MEMORY_DIR if compressed else MEMORY_DIR
    return base_dir / f"{user_id}_memory.pkl"

class TransferState(BaseModel):
    """Model to track the state of a transfer conversation."""
    step: str = Field(default="initial", description="Current step in the transfer flow")
    recipient_name: Optional[str] = Field(default=None, description="Name of the recipient being searched")
    amount: Optional[str] = Field(default=None, description="Amount to transfer")
    recipient_id: Optional[str] = Field(default=None, description="Selected recipient ID")
    description: Optional[str] = Field(default=None, description="Transfer description")
    category: Optional[str] = Field(default=None, description="Transfer category")
    reference: Optional[str] = Field(default=None, description="Transfer reference")
    scheduled_at: Optional[str] = Field(default=None, description="Scheduled transfer time")
    found_recipients: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of found recipients")
    needs_authorization: bool = Field(default=False, description="Whether transfer needs authorization")
    can_proceed: bool = Field(default=False, description="Whether the transfer can proceed with current details")

    @field_validator('found_recipients', mode='before')
    @classmethod
    def validate_found_recipients(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate the found_recipients field."""
        if v is not None and not isinstance(v, list):
            raise ValueError("found_recipients must be a list")
        return v

    def update_can_proceed(self) -> None:
        """Update whether the transfer can proceed based on current state."""
        self.can_proceed = bool(
            self.recipient_id and 
            self.amount and 
            self.step != "initial" and 
            self.found_recipients is not None and 
            len(self.found_recipients) > 0
        )

def get_transfer_state(user_id: str) -> TransferState:
    """Get the transfer state from cache."""
    state_data = cache.get(f"transfer_state_{user_id}")
    if state_data:
        return TransferState(**json.loads(state_data))
    return TransferState()

def save_transfer_state(user_id: str, state: TransferState) -> None:
    """Save the transfer state to cache."""
    cache.set(
        f"transfer_state_{user_id}",
        json.dumps(state.dict()),
        timeout=settings.SESSION_COOKIE_AGE
    )

def get_agent_response(user_input: str, user_id: str, tx_history: Optional[List[Dict[str, Any]]] = None, access_token: Optional[str] = None) -> str:
    """Get a response from the agent for a user's input."""
    try:
        # Initialize vector store tool with recent tx_history
        vector_store_tool = VectorStoreQueryTool(
            user_id=user_id,
            tx_history=tx_history or []  # Recent transactions passed to API
        )
        
        # Get chat history from FAISS
        try:
            chat_history = vector_store_tool.get_chat_history()
        except Exception as e:
            logger.error(f"Error getting chat history for user {user_id}: {e}")
            chat_history = []
            
        # Get historical transactions from FAISS
        try:
            faiss_tx_history = vector_store_tool.get_transaction_history()
            logger.info(f"Retrieved {len(faiss_tx_history)} historical transactions from FAISS for user {user_id}")
        except Exception as e:
            logger.error(f"Error getting transaction history from FAISS for user {user_id}: {e}")
            faiss_tx_history = []
            
        # Combine recent and historical transactions, with recent ones taking precedence
        combined_tx_history = []
        
        # Add historical transactions from FAISS first (older data)
        if faiss_tx_history:
            combined_tx_history.extend(faiss_tx_history)
            
        # Add recent transactions passed to API (newer data, overwrites older versions)
        if tx_history:
            # Create a set of transaction IDs we've already seen
            seen_tx_ids = {tx.get('id') for tx in combined_tx_history if tx.get('id')}
            
            # Only add transactions we haven't seen before
            for tx in tx_history:
                if tx.get('id') not in seen_tx_ids:
                    combined_tx_history.append(tx)
                    seen_tx_ids.add(tx.get('id'))
        
        logger.info(f"Combined {len(combined_tx_history)} total transactions for user {user_id}")
        
        # Create memory from chat history if it exists
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        if chat_history:
            logger.info(f"Found existing chat history for user {user_id}, using as context")
            for msg in chat_history:
                if msg["role"] == "user":
                    memory.save_context({"input": msg["content"]}, {"output": ""})
                elif msg["role"] == "assistant":
                    memory.save_context({"input": ""}, {"output": msg["content"]})
        else:
            logger.info(f"No existing chat history for user {user_id}, starting fresh conversation")
        
        # Update vector store tool with combined transaction history
        vector_store_tool.tx_history = combined_tx_history
        
        # Initialize transfer state
        transfer_state = TransferState()
        
        # Create tools with access token
        tools = [
            vector_store_tool,
            search_tool,
            GetSimilarRecipientsTool(),
            MakeTransferTool(),
            SignalTransferSuccessTool()
        ]
        
        # Set access token in environment for tools that need it
        if access_token:
            os.environ["ACCESS_TOKEN"] = access_token
        else:
            # Clear any existing token
            os.environ.pop("ACCESS_TOKEN", None)
        
        # Update transfer state's can_proceed flag
        transfer_state.update_can_proceed()
        
        # Create the agent with transfer state awareness
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            system_message=system_prompt + f"\nCurrent transfer state: {transfer_state.dict()}"
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True
        )
        
        # Get the response
        response = agent_executor.invoke({"input": user_input})
        
        # Update transfer state based on the response
        if "AUTHORIZE" in user_input.upper() and transfer_state.needs_authorization:
            # Reset transfer state after successful authorization
            transfer_state = TransferState()
        elif "transfer" in user_input.lower() and transfer_state.step == "initial":
            # Start new transfer flow
            transfer_state.step = "getting_recipient"
        elif "no recipients found" in response["output"].lower():
            # Reset transfer state if no recipients found
            transfer_state = TransferState()
        
        # Update can_proceed flag again after state changes
        transfer_state.update_can_proceed()
        
        # Save the current exchange to chat history
        try:
            vector_store_tool.save_to_chat_history(
                user_input=user_input,
                assistant_response=response["output"]
            )
            logger.info(f"Successfully saved chat exchange to history for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving chat history for user {user_id}: {e}")
            # Continue even if saving history fails - don't block the response
        
        return response["output"]
        
    except Exception as e:
        logger.error(f"Error in get_agent_response for user {user_id}: {e}", exc_info=True)
        return f"I apologize, but I encountered an error while processing your request. Please try again later. Error: {str(e)}"

def save_memory(user_id: str, memory: ConversationBufferMemory, compress: bool = False) -> None:
    """Save a user's memory to disk, optionally with compression."""
    try:
        memory_path = get_memory_path(user_id, compressed=compress)
        
        with open(memory_path, 'wb') as f:
            pickle.dump(memory, f)
        
        if compress:
            # Compress the file
            with open(memory_path, 'rb') as f_in:
                with gzip.open(f"{memory_path}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove the uncompressed file
            memory_path.unlink()
            
        logger.info(f"Saved {'compressed ' if compress else ''}memory for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving memory for user {user_id}: {e}")

def load_memory(user_id: str) -> ConversationBufferMemory:
    """Load a user's memory from disk, trying compressed version first."""
    try:
        # Try compressed version first
        compressed_path = get_memory_path(user_id, compressed=True)
        if compressed_path.with_suffix('.gz').exists():
            with gzip.open(compressed_path.with_suffix('.gz'), 'rb') as f:
                memory = pickle.load(f)
            logger.info(f"Loaded compressed memory for user {user_id}")
            return memory
            
        # Try uncompressed version
        memory_path = get_memory_path(user_id)
        if memory_path.exists():
            with open(memory_path, 'rb') as f:
                memory = pickle.load(f)
            logger.info(f"Loaded memory for user {user_id}")
            return memory
            
        # Create new memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory
    except Exception as e:
        logger.error(f"Error loading memory for user {user_id}: {e}")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory

def merge_chat_histories(user_id: str, histories: List[List[Dict[str, str]]], deduplicate: bool = True) -> None:
    try:
        memory = load_memory(user_id)
        messages = []
        
        # Convert existing messages
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add new messages
        for history in histories:
            if isinstance(history, list):
                messages.extend([msg for msg in history if isinstance(msg, dict) and "role" in msg and "content" in msg])
        
        # Deduplicate if needed
        if deduplicate:
            seen = set()
            messages = [msg for msg in messages if not (msg_key := f"{msg['role']}:{msg['content']}") in seen and not seen.add(msg_key)]
        
        # Create new memory with all messages
        new_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in messages:
            if msg["role"] == "user":
                new_memory.save_context({"input": msg["content"]}, {"output": ""})
            elif msg["role"] == "assistant":
                new_memory.save_context({"input": ""}, {"output": msg["content"]})
        
        save_memory(user_id, new_memory)
        logger.info(f"Merged {len(histories)} histories for user {user_id}")
    except Exception as e:
        logger.error(f"Error merging chat histories for user {user_id}: {e}")
        raise

def search_chat_history(
    user_id: str,
    query: str,
    max_results: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    message_type: Optional[Literal["user", "assistant", "all"]] = "all"
) -> List[Dict[str, Any]]:
    """Search through chat history with advanced filtering options."""
    try:
        memory = load_memory(user_id)
        results = []
        
        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        for i, message in enumerate(memory.chat_memory.messages):
            # Skip if message type doesn't match
            if message_type != "all":
                if message_type == "user" and not isinstance(message, HumanMessage):
                    continue
                if message_type == "assistant" and not isinstance(message, AIMessage):
                    continue
            
            # Get message timestamp
            msg_timestamp = message.additional_kwargs.get("timestamp")
            if msg_timestamp:
                msg_dt = datetime.fromisoformat(msg_timestamp)
                # Skip if outside date range
                if start_dt and msg_dt < start_dt:
                    continue
                if end_dt and msg_dt > end_dt:
                    continue
            
            # Check content
            if query.lower() in message.content.lower():
                result = {
                    "index": i,
                    "role": "user" if isinstance(message, HumanMessage) else "assistant",
                    "content": message.content,
                    "timestamp": msg_timestamp or "unknown"
                }
                results.append(result)
                if len(results) >= max_results:
                    break
        
        return results
    except Exception as e:
        logger.error(f"Error searching chat history for user {user_id}: {e}")
        raise

def backup_all_memories() -> str:
    """Create a backup of all memories."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"memory_backup_{timestamp}.tar.gz"
        
        with tarfile.open(backup_path, "w:gz") as tar:
            # Add all memory files
            for memory_file in MEMORY_DIR.glob("*_memory*"):
                tar.add(memory_file, arcname=memory_file.name)
            # Add all compressed files
            for compressed_file in COMPRESSED_MEMORY_DIR.glob("*_memory*"):
                tar.add(compressed_file, arcname=compressed_file.name)
        
        logger.info(f"Created backup at {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"Error creating memory backup: {e}")
        raise

def restore_from_backup(backup_path: str) -> None:
    """Restore memories from a backup file."""
    try:
        if not Path(backup_path).exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        with tarfile.open(backup_path, "r:gz") as tar:
            # Extract to temporary directory first
            temp_dir = MEMORY_DIR / "temp_restore"
            temp_dir.mkdir(exist_ok=True)
            tar.extractall(path=temp_dir)
            
            # Move files to appropriate directories
            for file in temp_dir.glob("*_memory*"):
                if file.suffix == '.gz':
                    shutil.move(file, COMPRESSED_MEMORY_DIR / file.name)
                else:
                    shutil.move(file, MEMORY_DIR / file.name)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
        
        logger.info(f"Restored memories from backup {backup_path}")
    except Exception as e:
        logger.error(f"Error restoring from backup {backup_path}: {e}")
        raise

def cleanup_old_memories(user_id: Optional[str] = None, days_threshold: int = MEMORY_EXPIRY_DAYS) -> None:
    """Remove memory files for inactive users."""
    try:
        current_time = datetime.now()
        base_dirs = [MEMORY_DIR, COMPRESSED_MEMORY_DIR]
        
        for base_dir in base_dirs:
            pattern = f"{user_id}_memory*" if user_id else "*_memory*"
            for memory_file in base_dir.glob(pattern):
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(memory_file.stat().st_mtime)
                    if current_time - mtime > timedelta(days=days_threshold):
                        memory_file.unlink()
                        logger.info(f"Removed old memory file: {memory_file}")
                except Exception as e:
                    logger.error(f"Error processing memory file {memory_file}: {e}")
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")

def compress_memory(user_id: str) -> None:
    """Compress a user's memory to reduce disk usage."""
    try:
        memory = load_memory(user_id)
        save_memory(user_id, memory, compress=True)
        logger.info(f"Compressed memory for user {user_id}")
    except Exception as e:
        logger.error(f"Error compressing memory for user {user_id}: {e}")
        raise

def export_chat_history(user_id: str, format: str = "json") -> Any:
    """Export a user's chat history in the specified format."""
    try:
        memory = load_memory(user_id)
        messages = []
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "assistant", "content": message.content})
        
        if format.lower() == "json":
            return json.dumps(messages, indent=2)
        elif format.lower() == "txt":
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Error exporting chat history for user {user_id}: {e}")
        raise

def import_chat_history(user_id: str, history: List[Dict[str, str]]) -> None:
    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] == "user":
                    memory.save_context({"input": msg["content"]}, {"output": ""})
                elif msg["role"] == "assistant":
                    memory.save_context({"input": ""}, {"output": msg["content"]})
        save_memory(user_id, memory)
        logger.info(f"Imported chat history for user {user_id}")
    except Exception as e:
        logger.error(f"Error importing chat history for user {user_id}: {e}")
        raise

def trim_memory(memory: ConversationBufferMemory) -> None:
    """Trim memory to prevent it from growing too large."""
    if len(memory.chat_memory.messages) > MAX_MEMORY_SIZE:
        # Keep the most recent messages
        memory.chat_memory.messages = memory.chat_memory.messages[-MAX_MEMORY_SIZE:]
        logger.info(f"Trimmed memory to {MAX_MEMORY_SIZE} messages")

# Initialize global memory store for each user
user_memories = {}

def list_available_backups() -> List[Dict[str, Any]]:
    """List all available backup files with their details."""
    try:
        backups = []
        for backup_file in BACKUP_DIR.glob("memory_backup_*.tar.gz"):
            try:
                stats = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size_bytes": stats.st_size,
                    "created_at": datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "modified_at": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                })
            except Exception as e:
                logger.error(f"Error processing backup file {backup_file}: {e}")
        
        # Sort by creation date, newest first
        backups.sort(key=lambda x: x["created_at"])
        return backups
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise

def schedule_automatic_backup(schedule_time: str, days: Optional[List[int]] = None) -> None:
    """Schedule automatic backups at specified time and days."""
    try:
        # Load existing schedule
        schedule_data = {}
        if BACKUP_SCHEDULE_FILE.exists():
            with open(BACKUP_SCHEDULE_FILE, 'r') as f:
                schedule_data = json.load(f)
        
        # Update schedule
        schedule_data["time"] = schedule_time
        schedule_data["days"] = days or list(range(7))  # Default to every day
        
        # Save schedule
        with open(BACKUP_SCHEDULE_FILE, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        # Clear existing schedule
        backup_scheduler.clear()
        
        # Add new schedule
        for day in schedule_data["days"]:
            backup_scheduler.every().day.at(schedule_time).do(backup_all_memories)
        
        # Start scheduler if not running
        global scheduler_thread
        if scheduler_thread is None or not scheduler_thread.is_alive():
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
        
        logger.info(f"Scheduled backups for {schedule_time} on days {schedule_data['days']}")
    except Exception as e:
        logger.error(f"Error scheduling automatic backup: {e}")
        raise

def run_scheduler() -> None:
    """Run the backup scheduler in a separate thread."""
    while True:
        backup_scheduler.run_pending()
        time.sleep(60)  # Check every minute

def get_backup_schedule() -> Dict[str, Any]:
    """Get the current backup schedule."""
    try:
        if BACKUP_SCHEDULE_FILE.exists():
            with open(BACKUP_SCHEDULE_FILE, 'r') as f:
                return json.load(f)
        return {"time": None, "days": []}
    except Exception as e:
        logger.error(f"Error reading backup schedule: {e}")
        raise

class ChatbotAgent:
    def __init__(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        tx_history: List[Dict[str, Any]],
        api_key: str,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # Updated default model
    ):
        """Initialize the chatbot agent with user context and tools."""
        self.user_id = user_id
        self.user_profile = user_profile
        self.tx_history = tx_history
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize transfer state
        self.transfer_state = TransferState()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize the agent
        self.agent_executor = self._initialize_agent()
        
        # Initialize vector store query tool for history
        self.history_tool = VectorStoreQueryTool(
            user_id=user_id,
            tx_history=tx_history
        )

    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools."""
        return [
            Tool(
                name="query_user_context",
                func=self.history_tool.run,
                description="Search the user's specific financial documents and chat history"
            ),
            Tool(
                name="get_similar_recipients",
                func=GetSimilarRecipientsTool().run,
                description="Get a list of similar recipients by name to transfer money to"
            ),
            Tool(
                name="make_transfer",
                func=MakeTransferTool().run,
                description="Transfers money to a specified recipient after gathering all necessary details"
            ),
            Tool(
                name="signal_transfer_success",
                func=SignalTransferSuccessTool().run,
                description="Signals the frontend that a transfer was successful"
            ),
            Tool(
                name="duckduckgo_search",
                func=search_tool.run,
                description="Search the web for current information"
            )
        ]

    def _initialize_agent(self) -> AgentExecutor:
        """Initialize the agent with tools and prompt."""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + f"\nCurrent transfer state: {self.transfer_state.dict()}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Initialize the language model with Groq
        llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0
        )

        # Create the agent using structured chat agent
        agent = StructuredChatAgent.from_llm_and_tools(
            llm=llm,
            tools=self.tools,
            system_message=system_prompt + f"\nCurrent transfer state: {self.transfer_state.dict()}"
        )

        # Create the agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def process_message(self, message: str) -> str:
        """Process a user message and return a response."""
        try:
            # Get chat history
            chat_history = self.history_tool.get_chat_history()
            
            # Get transaction history
            tx_history = self.history_tool.get_transaction_history()
            
            # Update the agent's context with the latest history
            self.tx_history = tx_history
            
            # Process the message
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })
            
            # Save the exchange to chat history
            self.history_tool.save_to_chat_history(message, response["output"])
            
            return response["output"]
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def save_transaction(self, transaction: Dict[str, Any]) -> None:
        """Save a transaction to the transaction history."""
        try:
            self.history_tool.save_to_transaction_history(transaction)
            self.tx_history.append(transaction)
        except Exception as e:
            logger.error(f"Error saving transaction: {e}")
            raise 
