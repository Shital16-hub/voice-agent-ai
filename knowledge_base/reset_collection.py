# Create a script to reset the collection
import asyncio
from knowledge_base.vector_store import ChromaStore

async def reset_collection():
    store = ChromaStore()
    await store.init()
    success = await store.reset_collection()
    print(f"Collection reset: {success}")

asyncio.run(reset_collection())