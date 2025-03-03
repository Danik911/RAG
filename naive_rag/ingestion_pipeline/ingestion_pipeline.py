#%%
%%capture
!pip install llama-index==0.10.37 cohere==5.5.0 openai==1.30.1 llama-index-embeddings-openai==0.1.9 qdrant-client==1.9.1 llama-index-vector-stores-qdrant==0.2.8 llama-index-llms-cohere==0.2.0 pydantic==2.0.
#%%
pip install --upgrade llama-index llama-index-llms-cohere llama-index-embeddings-openai
#%%
pip install pydantic==1.10.12
#%%
import os
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()

load_dotenv()

#%%
API_KEY_CO= os.getenv("API_KEY_COHERE")
API_KEY_OPEN_AI = os.getenv("API_KEY_OPEN_AI")
API_KEY_QDRANT = os.getenv("API_KEY_QDRANT")
QDRANT_CLIENT = os.getenv("QDRANT_CLIENT")

#%%
pip3 install pydantic==2.0.3
#%%
try:
    import pydantic.v1 as pydantic  # Update for older pydantic compatibility (if using v1.x)
    from pydantic.v1 import (
        BaseConfig,
        BaseModel,
        Field,
        create_model,
        root_validator,
        validator,
    )
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        ConfigDict,  # New in v2.x
        create_model,
        field_validator,  # New in v2.x (replaces `@validator`)
    )

# Update code to remove 'BaseConfig' dependency
BaseConfig = None  # If unused; otherwise, adjust logic where this class is needed

#%%
from llama_index.legacy.vector_stores import QdrantVectorStore
from qdrant_client import QdrantClient
from naive_rag.helpers.utils import setup_llm, setup_embed_model, setup_vector_store

# Initialize client
client = QdrantClient(url=QDRANT_CLIENT, api_key=API_KEY_QDRANT)

# Set up the vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="RAG_PROJECT",
    enable_hybrid=False  # Modify this to match your use case
)


setup_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key=API_KEY_OPEN_AI
    )

setup_embed_model(
    provider="openai",
    api_key=API_KEY_OPEN_AI
    )

vector_store = setup_vector_store(QDRANT_CLIENT, API_KEY_QDRANT, COLLECTION_NAME)
#%%
