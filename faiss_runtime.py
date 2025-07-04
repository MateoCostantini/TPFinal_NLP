import faiss
import numpy as np
import pickle
from openai import AzureOpenAI
import instructor
import os


class SQLiteFAISSRuntime:
    def __init__(
        self,
        DB_folder,
        azure_api_key: str,
        azure_api_base: str,
        azure_api_version: str,
        embedding_deployment: str,
        
    ):
        self.DB_folder = DB_folder
        self.index = faiss.read_index(os.path.join(self.DB_folder, "index_all.index"))
        with open(os.path.join(self.DB_folder, "mapping_all.pkl"), "rb") as f:
            self.mapping = pickle.load(f)
        with open(os.path.join(self.DB_folder, "inverted_index.pkl"), "rb") as f:
            self.inverted_index = pickle.load(f)

        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_api_base
        )
        self.client = instructor.from_openai(azure_client)


        self.embedding_deployment = embedding_deployment

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=query,
            model=self.embedding_deployment
        )

        q_emb = np.array(response.data[0].embedding).astype("float32")
        q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)

        return q_emb



    def search(self, word: str, table: str, column: str, k: int = 5, threshold: float = 0.65 ): 
        """
        Busca en la columna especificada los embeddings más parecidos
        a la palabra dada, devolviendo tupla (word, similarity, (tabla, columna, texto_original)).
        """
        q_emb = self.embed_query(word)
        subset_indices = self.inverted_index.get((table, column), [])
        if not subset_indices:
            return []

        # Reconstruir y normalizar vectores de subset
        all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
        subset_vecs = all_vecs[subset_indices]
        faiss.normalize_L2(subset_vecs)

        # FAISS index para subset
        subset_idx = faiss.IndexFlatIP(subset_vecs.shape[1])
        subset_idx.add(subset_vecs)

        # Buscar
        D, I = subset_idx.search(q_emb, k)

        results = []
        for sim, rel_idx in zip(D[0], I[0]):
            if sim >= threshold:
                global_idx = subset_indices[rel_idx]
                tabla, col, txt = self.mapping[global_idx]
                results.append((word, float(sim), (tabla, col, txt)))

        return results


