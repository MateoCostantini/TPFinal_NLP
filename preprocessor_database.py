import sqlite3
import faiss
import numpy as np
from openai import AzureOpenAI
import instructor
import os
import pickle
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import asyncio
import json



class SQLitePreprocessor:
    def __init__(
        self,
        sqlite_path: str,
        azure_api_key: str,
        azure_api_base: str,
        azure_api_version: str,
        embedding_deployment: str,
    ):
        self.sqlite_path = sqlite_path
        self.DB_folder = os.path.dirname(sqlite_path)
        self.index_path = os.path.join(self.DB_folder, "index_all.index")
        self.mapping_path = os.path.join(self.DB_folder, "mapping_all.pkl")
        self.inverted_index_path = os.path.join(self.DB_folder, "inverted_index.pkl")
        self.embedding_deployment = embedding_deployment
        
        self.schema_file = os.path.join(self.DB_folder, "schema.json")


        # Azure OpenAI setup
        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_api_base
        )
        self.client = instructor.from_openai(azure_client)

    def get_tables(self, cursor) -> List[str]:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def get_text_columns(self, cursor, table: str) -> List[str]:
        cursor.execute(f"PRAGMA table_info({table})")
        return [
            col[1] for col in cursor.fetchall()
            if any(tok in (col[2] or "").upper() for tok in ("CHAR", "CLOB", "TEXT", "VARCHAR"))
        ]

    def extract_texts_and_mapping(self, conn) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[Tuple[str, str], List[int]]]:
        cursor = conn.cursor()
        tables = self.get_tables(cursor)

        texts = []
        mapping = []
        inverted_index = defaultdict(list)

        for table in tables:
            text_columns = self.get_text_columns(cursor, table)
            if not text_columns:
                continue

            cursor.execute(f"SELECT {', '.join(text_columns)} FROM {table}")
            rows = cursor.fetchall()

            for row in rows:
                for idx, col_name in enumerate(text_columns):
                    text = row[idx]
                    if text and text.strip():
                        texts.append(text)
                        mapping.append((table, col_name, text))
                        inverted_index[(table, col_name)].append(len(mapping) - 1)

        return texts, mapping, dict(inverted_index)

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch embeddings"):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_deployment
            )
            embeddings.extend(r.embedding for r in response.data)
        return np.array(embeddings, dtype="float32")

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def save_all(self, index: faiss.Index, mapping: List[Tuple[str, str, str]], inverted_index: Dict[Tuple[str, str], List[int]]):
        faiss.write_index(index, self.index_path)
        with open(self.mapping_path, "wb") as f:
            pickle.dump(mapping, f)
        with open(self.inverted_index_path, "wb") as f:
            pickle.dump(inverted_index, f)

    def run(self):
        conn = sqlite3.connect(self.sqlite_path)
        try:
            texts, mapping, inverted_index = self.extract_texts_and_mapping(conn)
            if not texts:
                raise ValueError("No text columns found.")

            embeddings = self.generate_embeddings(texts)
            index = self.build_faiss_index(embeddings)
            self.save_all(index, mapping, inverted_index)
        finally:
            conn.close()




    def create_scheme(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"tables": {}}
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [r[0] for r in cur.fetchall()]

            for tbl in tables:
                cur.execute(f"PRAGMA table_info({tbl});")
                cols = cur.fetchall()

                columns = [{"name": c[1], "type": c[2]} for c in cols]
                primary_keys = [c[1] for c in cols if c[5] != 0]

                cur.execute(f"SELECT * FROM {tbl} LIMIT 1;")
                row = cur.fetchone()
                sample = None
                if row:
                    keys = [c["name"] for c in columns]
                    sample = dict(zip(keys, row))

                report["tables"][tbl] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "sample_row": sample
                }
        
        with open(self.schema_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)







