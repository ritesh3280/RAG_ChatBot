import os
import time
import json
from dotenv import load_dotenv
from embeddings_langchain import get_embeddings_with_cleaning, getText, chunk_splitters, clean_text
from pinecone import Pinecone, ServerlessSpec
from embeddings_langchain import process_document

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

documents_dir = "/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/"
processed_files_path = "processed_files.json"
cloud = 'aws'
region = 'us-east-1'
index_name = "langchain-rag"

def load_processed_files():
    """Load the list of processed files and their namespaces"""
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            return json.load(f)
    return {'files': {}, 'next_namespace': 0}

def save_processed_files(processed_data):
    """Save the updated list of processed files"""
    with open(processed_files_path, 'w') as f:
        json.dump(processed_data, f)

def verify_namespace_exists(index, namespace):
    """Verify if a namespace exists in Pinecone"""
    try:
        # Try to get stats for the namespace
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        return namespace in namespaces
    except Exception as e:
        print(f"Error verifying namespace: {str(e)}")
        return False

def cleanup_processed_files():
    """Remove records of files whose namespaces don't exist in Pinecone"""
    index = pc.Index(index_name)
    processed_data = load_processed_files()
    files_to_remove = []
    
    for filename, data in processed_data['files'].items():
        if not verify_namespace_exists(index, data['namespace']):
            files_to_remove.append(filename)
    
    for filename in files_to_remove:
        del processed_data['files'][filename]
        print(f"Removed record for {filename} as namespace no longer exists")
    
    save_processed_files(processed_data)

def initialize_pinecone():
    """Initialize Pinecone index if it doesn't exist"""
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} created successfully")
    
    # Clean up any stale records
    cleanup_processed_files()
    return pc.Index(index_name)

def upsert_single_document(filename):
    index = initialize_pinecone()
    processed_data = load_processed_files()
    
    namespace = f"document_{processed_data['next_namespace']}"
    doc_path = os.path.join(documents_dir, filename)
    
    try:
        embeddings, text_chunks = process_document(doc_path)
        
        # Create vectors with proper metadata
        upsert_data = []
        for i in range(len(embeddings)):
            vector_data = {
                "id": f"{namespace}_{i}",
                "values": embeddings[i],
                "metadata": {
                    "text": text_chunks[i],
                    "chunk_index": i,
                    "filename": filename
                }
            }
            upsert_data.append(vector_data)
        
        # Batch upsert in smaller chunks to avoid timeout
        batch_size = 100
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        processed_data['files'][filename] = {
            'namespace': namespace,
            'timestamp': time.time(),
            'num_vectors': len(upsert_data)
        }
        processed_data['next_namespace'] += 1
        save_processed_files(processed_data)
        
        print(f"Successfully upserted {filename} with {len(upsert_data)} vectors to namespace {namespace}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise



def search_pinecone(query_embedding, top_k=5):
    """Search across all namespaces in Pinecone index"""
    index = pc.Index(index_name)
    
    # Get all namespaces
    processed_data = load_processed_files()
    namespaces = [data['namespace'] for data in processed_data['files'].values()]
    
    # Search in each namespace and combine results
    all_results = []
    for namespace in namespaces:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        all_results.extend(results['matches'])
    
    # Sort by score and get top_k results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return {'matches': all_results[:top_k]}

def delete_document(filename):
    """Delete a document's vectors from its namespace"""
    processed_data = load_processed_files()
    if filename in processed_data['files']:
        namespace = processed_data['files'][filename]['namespace']
        index = pc.Index(index_name)
        index.delete(deleteAll=True, namespace=namespace)
        del processed_data['files'][filename]
        save_processed_files(processed_data)
        print(f"Deleted document {filename} from namespace {namespace}")