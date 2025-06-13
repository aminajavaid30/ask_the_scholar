from langchain_community.vectorstores import Chroma
from ingest import JinaCLIPLangchainWrapper
from transformers import AutoModel
from logger import logger
from dotenv import load_dotenv
load_dotenv()

class Retrieval:
    def __init__(self, persist_directory="./chroma_db"):
        raw_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        self.embedding_model = JinaCLIPLangchainWrapper(raw_model)

        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embedding_model,
            persist_directory=persist_directory
        )

    def retrieve(self, query: str, k: int = 5):
        """Retrieve relevant documents based on the query."""
        logger.info(f"[Retrieve] Retrieving documents for query: {query}")
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            logger.warning("[Retrieve] No relevant documents found.")
        return results
    
    def get_vector_store(self):
        return self.vector_store
