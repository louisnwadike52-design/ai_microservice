# Force Python to use the pysqlite3 library instead of the system one.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Using pysqlite3-binary for sqlite3")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3. This might cause issues.")
    pass

import chromadb
import os
import argparse

# --- Configuration (match your services.py) ---
# You might want to make these command-line arguments
DEFAULT_CHROMA_DATA_PATH = "chroma_data/"
DEFAULT_COLLECTION_NAME = "user_transactions"

def inspect_chromadb(chroma_path: str, collection_name: str, limit: int):
    """Connects to ChromaDB and inspects the specified collection."""

    abs_chroma_path = os.path.abspath(chroma_path)
    print(f"Attempting to connect to ChromaDB at: {abs_chroma_path}")

    if not os.path.exists(abs_chroma_path):
        print(f"Error: Chroma path '{abs_chroma_path}' does not exist.")
        return

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        print(f"Successfully connected to ChromaDB.")

        # --- List existing collections ---
        collections = client.list_collections()
        print("\nExisting Collections:")
        if not collections:
            print("- No collections found.")
            return # Exit if no collections
        else:
            collection_names = [coll.name for coll in collections]
            for coll in collections:
                print(f"- Name: {coll.name}, Metadata: {coll.metadata}")

            if collection_name not in collection_names:
                 print(f"\nWarning: Specified collection '{collection_name}' not found in the database.")
                 # Optionally, list contents of the first collection found? Or just exit?
                 # Let's try accessing the first one if the default wasn't found and only one exists
                 if collection_name == DEFAULT_COLLECTION_NAME and len(collection_names) == 1:
                     collection_name = collection_names[0]
                     print(f"Attempting to inspect the first collection found: '{collection_name}'")
                 else:
                    print("Please specify a valid collection name using the --collection argument.")
                    return


        # --- Get the specified collection ---
        print(f"\nAccessing collection: '{collection_name}'")
        try:
            collection = client.get_collection(name=collection_name)

            # --- Get total count ---
            count = collection.count()
            print(f"Total items in collection '{collection_name}': {count}")

            # --- Peek at items ---
            if count > 0:
                actual_limit = min(limit, count) # Don't try to peek more than exists
                print(f"\nPeeking at up to {actual_limit} item(s) from '{collection_name}':")
                # Peek returns a limited number of items without needing specific IDs
                peek_result = collection.peek(limit=actual_limit)
                if peek_result and peek_result.get('ids'):
                    for i, doc_id in enumerate(peek_result['ids']):
                        print(f"  --- Item {i+1} ---")
                        print(f"    ID        : {doc_id}")
                        # Safely access documents and metadatas
                        metadata = peek_result.get('metadatas', [])[i] if peek_result.get('metadatas') and len(peek_result['metadatas']) > i else {}
                        doc_content = peek_result.get('documents', [])[i] if peek_result.get('documents') and len(peek_result['documents']) > i else None

                        print(f"    Metadata  : {metadata}")
                        if doc_content:
                           print(f"    Document  : {doc_content[:150] + ('...' if len(doc_content) > 150 else '')}") # Print start of document
                        else:
                            print("    Document  : [Not Available in Peek Result]") # Documents might not be included

                        # Embeddings are large, usually not useful to print directly
                        # embedding = peek_result.get('embeddings', [])[i] if peek_result.get('embeddings') and len(peek_result['embeddings']) > i else None
                        # print(f"    Embedding : {'Present' if embedding else 'N/A'}")

                else:
                    print("- Collection might be empty or peek result format unexpected.")
            else:
                 print("- Collection is empty.")

        except Exception as e:
            print(f"Error accessing collection '{collection_name}': {e}")
            import traceback
            traceback.print_exc()


    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB content.")
    parser.add_argument(
        "--path",
        default=DEFAULT_CHROMA_DATA_PATH,
        help=f"Path to the ChromaDB persistence directory (default: {DEFAULT_CHROMA_DATA_PATH})"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Name of the collection to inspect (default: {DEFAULT_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of items to peek from the collection (default: 5)"
    )

    args = parser.parse_args()

    inspect_chromadb(args.path, args.collection, args.limit) 