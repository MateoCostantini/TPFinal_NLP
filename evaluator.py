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
        self.db_path = db_path
        self.DB_folder = os.path.dirname(db_path)
        self.schema_path = os.path.join(self.DB_folder, "schema.json")
        self.schema = self._load_schema(self.schema_path)

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
            "- Use different tables and/or columns when possible, especially lesser-known or uncommon attributes."
        )

        resultados = []
        preguntas_vistas = set()

        while len(resultados) < N:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate a valid SQL query with its corresponding natural language question."}
                ],
                response_model=SQLconPregunta
            )

            result = res.model_dump()
            pregunta = result['pregunta']

            if pregunta not in preguntas_vistas:
                preguntas_vistas.add(pregunta)
                resultados.append(result)

        return resultados
    

    def evaluar_modelo_sql(self, modelo_generador, N = 20):
        """
        Evalúa un modelo generador de SQL a partir de preguntas.
        
        Args:
            modelo_generador: función que toma (schema, pregunta) y devuelve SQL.
            N: número de ejemplos a generar y evaluar.
        
        Returns:
            Lista de dicts con pregunta, SQL esperada, generada y similitud.
        """
        ground_truths = self.generar_sql_preguntas(N)
        resultados = []

        for gt in ground_truths:
            pregunta = gt['pregunta']
            sql_esperada = gt['sql']

            try:
                sql_generada = modelo_generador(pregunta)
            except Exception as e:
                sql_generada = f"[ERROR: {str(e)}]"

            similitud = difflib.SequenceMatcher(None, sql_esperada.lower(), sql_generada.lower()).ratio()

            resultados.append({
                'pregunta': pregunta,
                'sql_esperada': sql_esperada,
                'sql_generada': sql_generada,
                'similitud': similitud
            })

        return resultados
