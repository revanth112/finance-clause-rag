import json
from pathlib import Path
from app.retrieval.embeddings import BGEEmbeddingService
from app.retrieval.faiss_index import FAISSIndexService
from app.retrieval.metadata_store import MetadataStore


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.embedding_service = BGEEmbeddingService(model_name=model_name)
        self.index_service = None
        self.metadata_store = MetadataStore()

    def build_index_from_chunks(self, chunks_dir: str = "data/chunks"):
        chunk_files = list(Path(chunks_dir).glob("*.json"))
        all_chunks = []

        for file in chunk_files:
            chunk_data = json.loads(file.read_text(encoding="utf-8"))
            all_chunks.extend(chunk_data)

        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.index_service = FAISSIndexService(dim=embeddings.shape[1])
        self.index_service.add_embeddings(embeddings)
        self.metadata_store.add_records(all_chunks)

        self.index_service.save("data/vectorstore/faiss.index")
        self.metadata_store.save("data/vectorstore/chunk_metadata.json")

        return {
            "chunk_count": len(all_chunks),
            "embedding_dim": embeddings.shape[1]
        }

    def load_index(self):
        from app.retrieval.faiss_index import FAISSIndexService
        self.index_service = FAISSIndexService(dim=1024)  # placeholder not used after load
        self.index_service.index = FAISSIndexService.load("data/vectorstore/faiss.index")
        self.metadata_store = MetadataStore.load("data/vectorstore/chunk_metadata.json")

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_service.embed_query(query)
        scores, indices = self.index_service.search(query_embedding, top_k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            record = self.metadata_store.get_record(int(idx))
            results.append({
                "score": float(score),
                "chunk_id": record.get("chunk_id"),
                "doc_id": record.get("doc_id"),
                "text": record.get("text"),
                "section_title": record.get("section_title"),
                "clause_type": record.get("clause_type"),
                "metadata": record.get("metadata", {})
            })

        return results