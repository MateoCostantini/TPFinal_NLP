import sqlite3
import json
from typing import Dict, List, Optional, Any
from openai import AzureOpenAI
import instructor
import os
from pydantic import BaseModel, Field
import re
import sqlglot
from sqlglot import exp
import difflib
from tqdm import tqdm

class SQLconPregunta(BaseModel):
    sql: str = Field(description="The SQL query generated based on the database schema.")
    pregunta: str = Field(description="A natural language question that the randomly generated SQL query answers.")

class Evaluator:
    def __init__(
        self,
        db_path: str,
        azure_api_key: str,
        azure_api_endpoint: str,
        azure_api_version: str,
        deployment: str,
        faiss_runtime: Any = None
    ):
        self.db_path = "database\sakila_database\sakila_master.db"
        self.DB_folder = os.path.dirname(db_path)
        self.schema_path = os.path.join(self.DB_folder, "schema.json")
        self.schema = self._load_schema(self.schema_path)
        self.ground_truth_path = "database/evaluator/ground_truth.json"

        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_api_endpoint,
            api_version=azure_api_version
        )
        self.client = instructor.from_openai(azure_client)
        self.model = deployment

        self.faiss_runtime = faiss_runtime

    def _load_schema(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def normalizar_sql(self, sql: str) -> str:
        """
        Limpia una SQL para comparación:
        - Pasa a minúsculas.
        - Elimina aliases (AS ...).
        - Elimina espacios múltiples.
        - Elimina espacios alrededor de comillas.
        """
        sql = sql.lower()
        sql = re.sub(r"\s+as\s+\w+", "", sql)  # elimina alias tipo "as nombre"
        sql = re.sub(r"\s+", " ", sql)         # normaliza espacios
        sql = re.sub(r"\s*=\s*'?\s*(\d+)\s*'?", r"=\1", sql)  # ' 1' → 1
        return sql.strip()

        
    def generar_sql_preguntas(self, N: int) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a generator of SQL queries and their corresponding natural language questions.\n"
            "You will generate a syntactically correct SQL query that makes sense given the following database schema and its relationships:\n\n"
            f"{json.dumps(self.schema, indent=2)}\n\n"
            "Then, you will explain what the query does in a clear and concise natural language question (in Spanish).\n"
            "IMPORTANT:\n"
            "- The query must be syntactically correct, useful, and use real table and column names.\n"
            "- The question must accurately reflect what the SQL query is trying to retrieve.\n"
            "- Do NOT add any extra text—only the SQL query and the corresponding question.\n"
            "- EXTREMELY IMPORTANT: All queries must be different from each other.\n"
            "- Use different tables and/or columns when possible, you can use the sample_row from the database scheme to find attributes."
        )

        resultados = []
        preguntas_vistas = set()

        # Si ya existe, cargarlo para no repetir preguntas
        if os.path.exists(self.ground_truth_path):
            with open(self.ground_truth_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                preguntas_vistas = {d["pregunta"] for d in prev_data}
                resultados.extend(prev_data)
        else:
            prev_data = []
        prev_data_len = len(prev_data) 

        while len(resultados) < N + prev_data_len:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a new valid SQL query along with its corresponding natural language question." 
                     f" Make sure it is different from all previously generated SQL-question pairs listed below. \n\n {str(resultados)}"}
                ],
                response_model=SQLconPregunta
            )

            result = res.model_dump()
            pregunta = result['pregunta']

            if pregunta not in preguntas_vistas:
                preguntas_vistas.add(pregunta)
                resultados.append(result)
            
             # Guardar todo (original + nuevos) en el archivo
            os.makedirs(os.path.dirname(self.ground_truth_path), exist_ok=True)
            with open(self.ground_truth_path, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)

        print("success")
        return resultados
    

    def evaluar_modelo_sql(self, modelo_generador):
        """
        Evalúa un modelo generador de SQL a partir de preguntas usando un dataset ya generado.

        Args:
            modelo_generador: función que toma (pregunta) y devuelve una query SQL.
            dataset_path: ruta al archivo JSON con el dataset de ground truth.

        Returns:
            Lista de dicts con pregunta, SQL esperada, SQL generada y similitud.
        """
        

        if not os.path.exists(self.ground_truth_path):
            raise FileNotFoundError(f"El archivo de dataset '{self.ground_truth_path}' no existe.")

        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

        resultados = []

        for gt in tqdm(ground_truths, desc="Evaluando modelo SQL"):
            pregunta = gt['pregunta']
            sql_esperada = gt['sql']
            sql_esperada = self.normalizar_sql(sql_esperada)


            try:
                sql_generada = modelo_generador.generate_sql_with_embedded_variants(pregunta)
                sql_generada = self.normalizar_sql(sql_generada)
            except Exception as e:
                sql_generada = f"[ERROR: {str(e)}]"

            similitud = difflib.SequenceMatcher(None, sql_esperada.lower(), sql_generada.lower()).ratio()

            resultados.append({
                'pregunta': pregunta,
                'sql_esperada': sql_esperada,
                'sql_generada': sql_generada,
                'similitud': similitud
            })
            # Mostrar comparación por consola
            # print("\n---")
            # print(f"🟢 Pregunta:      {pregunta}")
            # print(f"✅ SQL esperada:  {sql_esperada}")
            # print(f"🛠️  SQL generada: {sql_generada}")
            # print(f"📊 Similitud:     {similitud:.2f}")


        return resultados
