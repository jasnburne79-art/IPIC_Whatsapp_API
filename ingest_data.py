#
# -------------------- ingest_data.py (v6 - Smarter Chunking) --------------------
#
import os
import re
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter # <-- IMPORT THIS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --- Configuration ---
SOURCE_DIRECTORY_PATH = "data/"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- Helper and Pre-processing Functions (Unchanged) ---
def calculate_checksum(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_processed_files_from_db(supabase: Client):
    try:
        response = supabase.table("ingestion_log").select("file_path, checksum").execute()
        return {item['file_path']: item['checksum'] for item in response.data}
    except Exception:
        return {}

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def standardize_terms(text):
    term_map = {
        r'\bplay park\b': 'IPIC Play', r'\bplay area\b': 'IPIC Play',
        r'\bgym\b': 'IPIC Active',
    }
    for old, new in term_map.items():
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text

def production_ingestion_pipeline():
    """
    An advanced pipeline that uses Markdown-aware chunking for better contextual retrieval.
    """
    print("Starting production ingestion pipeline...")
    # --- Connections (Unchanged) ---
    graph = Neo4jGraph()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    graph_generation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- File Tracking (Unchanged) ---
    print("\nStep 2: Checking for file changes...")
    processed_log = get_processed_files_from_db(supabase)
    current_files = {os.path.join(SOURCE_DIRECTORY_PATH, f): calculate_checksum(os.path.join(SOURCE_DIRECTORY_PATH, f))
                     for f in os.listdir(SOURCE_DIRECTORY_PATH) if os.path.isfile(os.path.join(SOURCE_DIRECTORY_PATH, f))}
    
    files_to_add = {f for f in current_files if f not in processed_log}
    files_to_delete = {f for f in processed_log if f not in current_files}
    files_to_update = {f for f in current_files if f in processed_log and current_files[f] != processed_log[f]}

    if not files_to_add and not files_to_delete and not files_to_update:
        print("✅ Knowledge base is already up-to-date.")
        return

    # --- Deletion Logic (Unchanged) ---
    files_requiring_deletion = files_to_delete.union(files_to_update)
    if files_requiring_deletion:
        print(f"\nStep 3: Deleting data for {len(files_requiring_deletion)} file(s)...")
        for file_path in files_requiring_deletion:
            graph.query("MATCH (s:Source {uri: $source_path})-[*0..]-(n) DETACH DELETE s, n", params={"source_path": file_path})
            supabase.table("documents").delete().eq("metadata->>source", file_path).execute()
            supabase.table("ingestion_log").delete().eq("file_path", file_path).execute()

    # --- Additions and Updates ---
    files_to_process = files_to_add.union(files_to_update)
    if files_to_process:
        print(f"\nStep 4: Processing {len(files_to_process)} file(s)...")
        
        all_chunks = []
        for file_path in files_to_process:
            with open(file_path, 'r') as f:
                markdown_text = f.read()
            
            # --- START: NEW MARKDOWN CHUNKING LOGIC ---
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            chunks = markdown_splitter.split_text(markdown_text)

            # Pre-process and add source metadata to each chunk
            for chunk in chunks:
                chunk.page_content = normalize_text(standardize_terms(chunk.page_content))
                chunk.metadata["source"] = file_path
            
            all_chunks.extend(chunks)
            # --- END: NEW MARKDOWN CHUNKING LOGIC ---

        print(f"Created {len(all_chunks)} document chunks from changed files.")

        # --- Graph Generation and Enrichment (Logic is the same, just uses the new chunks) ---
        llm_transformer = LLMGraphTransformer(
            llm=graph_generation_llm,
            allowed_nodes=["Policy", "Rule", "Membership", "Party", "Guest", "Item", "Payment", "Action", "Condition", "Location"],
            allowed_relationships=["APPLIES_TO", "CONCERNS", "PROHIBITS", "REQUIRES", "INCLUDES", "HAS_CONDITION", "MUST_PERFORM", "HAS_FEE"],
            strict_mode=True
        )
        
        all_graph_documents, enriched_chunks = [], []
        for chunk in all_chunks:
            graph_document = llm_transformer.convert_to_graph_documents([chunk])
            if graph_document:
                for node in graph_document[0].nodes:
                    node_text = f"A node representing a '{node.type}' named '{node.id}'."
                    node.properties["embedding"] = embeddings.embed_query(node_text)
                all_graph_documents.extend(graph_document)
                
                extracted_entities = [f"{node.type}:{node.id}" for node in graph_document[0].nodes]
                enriched_metadata = chunk.metadata.copy()
                enriched_metadata['graph_entities'] = extracted_entities
                enriched_chunks.append(Document(page_content=chunk.page_content, metadata=enriched_metadata))
        
        if all_graph_documents:
            graph.add_graph_documents(all_graph_documents, baseEntityLabel=True, include_source=True)
        if enriched_chunks:
            SupabaseVectorStore.from_documents(
                documents=enriched_chunks, embedding=embeddings, client=supabase,
                table_name="documents", query_name="match_documents"
            )

    # --- Update Log (Unchanged) ---
    print("\nStep 5: Updating database ingestion log...")
    for file_path in files_to_process:
        checksum = current_files[file_path]
        supabase.table("ingestion_log").upsert({"file_path": file_path, "checksum": checksum}).execute()
    
    print("\n✅ Production ingestion pipeline complete!")

if __name__ == "__main__":
    production_ingestion_pipeline()