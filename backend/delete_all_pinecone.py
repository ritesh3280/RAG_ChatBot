import os
from dotenv import load_dotenv
from pinecone import Pinecone

def delete_all_namespaces():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "langchain-rag"
    index = pc.Index(index_name)
    
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {}).keys()
        
        for namespace in namespaces:
            index.delete(deleteAll=True, namespace=namespace)
            print(f"Deleted namespace: {namespace}")
        
        print("All namespaces deleted successfully.")
    except Exception as e:
        print(f"Error deleting namespaces: {str(e)}")

if __name__ == "__main__":
    delete_all_namespaces()