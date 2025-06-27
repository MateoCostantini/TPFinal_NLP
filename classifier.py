from pydantic import BaseModel, ValidationError, Field
import json
from typing import Dict, List, Optional, Any, Literal
from openai import AzureOpenAI
import os
import instructor


class ClassifierInstructions(BaseModel):
        razonamiento: str = Field(
            description=(
                "Brief and clear explanation of why the category was chosen; "
                "mention tables/columns if applicable."
            )
        )
        category: Literal[
            "observes_database",
            "modifies_database"
        ]

class ClassifierService:
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
        

    def classify_query(self, user_query: str) -> str:
        system_prompt = f"""
        Eres un **clasificador experto en bases de datos**.
        Debes analizar la consulta del usuario y responder con un JSON en español
        que siga EXACTAMENTE esta estructura:

        {{
          "category": "<observes_database | modifies_database>",
          "razonamiento": "<explicación breve y clara de por qué elegiste la categoría>"
        }}

        #### Definiciones de las categorías
        - **observes_database**: la petición es solo de lectura (SELECT).
        - **modifies_database**: la petición implica inserciones, actualizaciones, eliminaciones o alteraciones de la estructura.

        Reglas:
        1. Razonamiento conciso en español; menciona tablas si aporta.
        2. Ignora errores tipográficos menores.
        3. Devuelve ÚNICAMENTE el JSON (sin texto extra).

        A continuación te paso las tablas y relaciones:

        {json.dumps(self.schema, indent=2, ensure_ascii=False)}
        """.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.0,
            response_model=ClassifierInstructions
        )

        return response.category