"""
memory_store.py

Thin wrapper around ChromaDB for the agent's long-term memory.
No security logic lives here on purpose — this is the "vulnerable
baseline" plumbing. Gate v1 / v2 will sit in gate.py, called from
agent.py BEFORE anything reaches write() here.
"""

import uuid
import time
import chromadb


class MemoryStore:
    def __init__(self, path: str = "./chroma_data", collection_name: str = "agent_memory"):
        # Persistent client -> memories survive across process restarts,
        # which matters for your "fresh session" trigger tests.
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def write(self, text: str, metadata: dict | None = None) -> str:
        """
        Write one memory item. Returns the generated memory id.

        metadata is where provenance/trust info will live once Gate v1
        is wired in (e.g. {"source": "user_direct", "trust": "high"}).
        For now it's optional and unused by anything downstream.
        """
        memory_id = str(uuid.uuid4())
        meta = metadata.copy() if metadata else {}
        meta.setdefault("written_at", time.time())

        self.collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[meta],
        )
        return memory_id

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Return the top-k memories most relevant to `query`.
        Each result: {"id": ..., "text": ..., "metadata": ..., "distance": ...}
        """
        if self.collection.count() == 0:
            return []

        k = min(k, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=k)

        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out

    def all_memories(self) -> list[dict]:
        """Dump everything — useful for debugging/logging trial runs."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get()
        out = []
        for i in range(len(results["ids"])):
            out.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return out

    def reset(self):
        """Wipe the collection clean. Use this between trial runs."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)


if __name__ == "__main__":
    # Quick manual smoke test — run `python memory_store.py` directly.
    store = MemoryStore(path="./chroma_data_test")
    store.reset()

    store.write("User's favorite color is blue.", {"source": "user_direct"})
    store.write("User works at a robotics startup.", {"source": "user_direct"})
    store.write("User mentioned they like hiking on weekends.", {"source": "user_direct"})

    results = store.retrieve("what does the user do for a living?", k=2)
    print("Query: what does the user do for a living?")
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['text']}  (meta={r['metadata']})")