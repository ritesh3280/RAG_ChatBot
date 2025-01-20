import os
import time
import json
from dotenv import load_dotenv
from embeddings_langchain import getText, chunk_splitters
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
    if os.path.exists(processed_files_path):
        with open(processed_files_path, 'r') as f:
            return json.load(f)
    return {'files': {}, 'next_namespace': 0}

def save_processed_files(processed_data):
    with open(processed_files_path, 'w') as f:
        json.dump(processed_data, f)

def verify_namespace_exists(index, namespace):
    try:

        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {})
        return namespace in namespaces
    except Exception as e:
        print(f"Error verifying namespace: {str(e)}")
        return False

def cleanup_processed_files():
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
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} created successfully")
    
    cleanup_processed_files()
    return pc.Index(index_name)

def upsert_single_document(filename):
    index = initialize_pinecone()
    processed_data = load_processed_files()
    
    namespace = f"document_{processed_data['next_namespace']}"
    doc_path = os.path.join(documents_dir, filename)
    
    try:
        embeddings, text_chunks = process_document(doc_path)
        # Printing processed for testing
        print(f"Processed {filename} with {len(embeddings)} vectors")
        
        upsert_data = []
        for i in range(len(embeddings)):
            flattened_embedding = embeddings[i] if isinstance(embeddings[i], list) and not isinstance(embeddings[i][0], list) else embeddings[i][0]
            vector_data = {
                "id": f"{namespace}_{i}",
                "values": flattened_embedding,
                "metadata": {
                    "filename": filename,
                    "person_name": text_chunks[0].split("\n")[1].strip(),
                    "text": text_chunks[i],
                    "chunk_index": i,
                    "section" : text_chunks[i].split("\n")[0].split(":")[1].strip(),
                }
            }
            # Printing upserting for testing
            print(f"Upserting vector {i} for {filename}")
            upsert_data.append(vector_data)
            # Printing upserted for testing
            print(f"Upserted vector {i} for {filename}")
        
        print("Length of upsert data : ", len(upsert_data))
        
        batch_size = 16
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i:i + batch_size]
            # Printing upserting batch for testing
            print(f"Upserting batch {i} to {i + batch_size} for {filename}")
            index.upsert(vectors=batch, namespace=namespace)
            # printing upserted for testing
            print(f"Upserted batch {i} to {i + batch_size} for {filename}")
            
        
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
    index = pc.Index(index_name)
    
    processed_data = load_processed_files()
    namespaces = [data['namespace'] for data in processed_data['files'].values()]
    
    all_results = []
    for namespace in namespaces:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        all_results.extend(results['matches'])
    
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return {'matches': all_results[:top_k]}

def delete_document(filename):
    processed_data = load_processed_files()
    if filename in processed_data['files']:
        namespace = processed_data['files'][filename]['namespace']
        index = pc.Index(index_name)
        index.delete(deleteAll=True, namespace=namespace)
        del processed_data['files'][filename]
        save_processed_files(processed_data)
        print(f"Deleted document {filename} from namespace {namespace}")