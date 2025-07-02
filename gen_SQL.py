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




class SQLResponse(BaseModel):
    razonamiento: str = Field( description = "Think clearly about what the system should do based on the user's query and the information in the database. Justify ALL the tables and columns it would need to access in order to respond.")
    SQL: str = Field(..., description="Generated SQL Query")

class SQLSelected(BaseModel):
    SQL: str = Field(..., description="Choose the SQL Query that best fits the user question")




class NLToSQLService:
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

    def choose_variants(self, question: str, queries: List[str]) -> str:
        prompt = (
            "You are an expert SQL assistant. "
            "You are given a user's question and a list of different SQL queries that could answer it. "
            "Your task is to carefully analyze which SQL query is the most correct, complete, and precise for answering the question. "
            "Return ONLY the chosen SQL query as plain text, with NO explanations, NO markdown, and NO extra text.\n\n"
            f"Question: {question}\n\n"
            "SQL Queries:\n"
        )

        sql_alternatives = ""
        for idx, q in enumerate(queries, 1):
            sql_alternatives += f"Query {idx}:\n{q}\n\n"
        sql_alternatives += "Return ONLY the best SQL query."

        resp = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sql_alternatives}
            ],
            model=self.model,
            max_tokens=800,
            temperature=0.0,
            response_model=SQLSelected,
        )
        sql = resp.SQL
        return sql
        

    def generate_sql(self, user_question, conversation = None) -> str:
        # if conversation == None:
        #     conversation = []

        schema_repr = json.dumps(self.schema, indent=2, ensure_ascii=False)
        prompt = (
            "You are an SQL expert managing a database for SQLite and you will assist by generating a query."
            "A user will ask you a question. Using the database structure,"
            "you must generate a single, correct SQL query to retrieve the requested information."
            "The resulting table must include, without exception, all the fields requested and relevant to the user's question."
            "IMPORTANT:"
            "ONLY return the SQL query (no explanations, no markdown, no extra text)."
            "The query must be syntactically correct."
            "If the question cannot be answered directly with the given tables, return an empty SELECT: SELECT 'Not available';"
            #"BUT: If the user asks a general question about what can be queried or about the structure of the database, return a sample query showing a few representative tables and columns."
            "The answer must be quick, efficient, and as direct as possible."
            #"Correctly consider which fields from the database the user needs and also include information that could be useful."
            "Ensure to show the columns that the user is asking for and additionaly, every column that you are using to compare with (==, LIKE, etc.) in the query."
            "Ensure that rows are not duplicated."
            "Use descriptive names for the generated columns."
            #"Always include all columns from the tables used in the query. Especially include ALL columns from each table you join or query."

            "\nBelow I will provide you with the tables and their relationships:\n\n" + schema_repr
        )

        # user_question = (
        #     f"Question: {question}\n\n"
        #     "Return only the corresponding SQL query (without explanations)."
        # )
     

        if conversation == None:
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_question}
                ] ,
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                response_model=SQLResponse,
            )
        else:
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                    #{"role": "user", "content": user_question}
                ] + conversation,
                model=self.model,
                max_tokens=512,
                temperature=0.0,
                response_model=SQLResponse,
            )
        sql = resp.SQL
        return sql
    

    def generate_sql_with_embedded_variants(self, question: str, conversation = None) -> str:
        """
        1 Genera SQL base
        2 Si es válida, parsea usando sqlglot para extraer todas las comparaciones de literales
        3 Para cada (tabla, columna, literal) usa FAISS para encontrar variantes
        4 Genera variantes reemplazando cada literal por alternativas FAISS
        5 Devuelve string con la query SQL.
        """
        if conversation == None:
            sql = self.generate_sql(question)
        else:
            sql = self.generate_sql(question, conversation)

        if sql.lower() == "select 'not available';":
            return sql

        try:
            parsed = sqlglot.parse_one(sql)
        except Exception as e:
            print(f"SQL parse error: {e}")
            return sql

        # 1. Resuelve alias -> tabla real
        alias_to_table = {}
        default_table = None

        for table in parsed.find_all(exp.Table):
            if table.alias:
                alias_to_table[table.alias] = table.name
                default_table = table.alias
            else:
                alias_to_table[table.name] = table.name
                default_table = table.name

        # 2. Encuentra comparaciones con literales
        replacements = []  # List[(original_literal, (table, column, value))]
        for where in parsed.find_all(exp.Where):
            for pred in where.find_all(exp.Predicate):
                if isinstance(pred, exp.EQ) or isinstance(pred, exp.Like):
                    left = pred.left
                    right = pred.right
                    if isinstance(right, exp.Literal) and isinstance(left, exp.Column):
                        col = left.name
                        tbl_or_alias = left.table or default_table  # 👈 usa default si no hay tabla
                        tbl = alias_to_table.get(tbl_or_alias, tbl_or_alias)
                        val = right.this
                        replacements.append((val, (tbl, col, val)))

        # No literales -> solo devuelve original
        if not replacements:
            return sql
        #print(replacements) 
        # 3 Busca variantes FAISS por cada literal
        replacements_map = {}  # val -> [alt1, alt2, ...]
        for val, (table, column, _) in replacements:
            hits = self.faiss_runtime.search(
                word=val.strip('%'),
                table=table,
                column=column,
                k=3,
                threshold=0.55
            )
            #print(f"faiss busca por: {val.strip('%')} en {table}.{column}")
            #print(hits)
            alt_texts = [h[2][2] for h in hits if h[2][2] != val]
            if alt_texts:
                replacements_map[val] = alt_texts
                #print(replacements_map[val])

        # 4 Genera variantes
        variants = [sql]  # original primero
        for val, alternatives in replacements_map.items():
            new_variants = []
            for base in variants:
                for alt in alternatives:
                    replaced = base.replace(f"'{val}'", f"'{alt}'")
                    new_variants.append(replaced)
            variants.extend(new_variants)
        
        if len(variants) > 1:
            variants.pop(0)

        # Quita duplicados
        variants = list(dict.fromkeys(variants))
        final_sql = self.choose_variants(question, variants)

        return final_sql


    # def generate_sql_with_embedded_variants(self, question: str) -> List[str]:
    #     """
    #     1 Genera la SQL base.
    #     2 Si es una SQL válida (no 'SELECT Not available'), busca todos los '... = 'valor''.
    #     3 Para cada valor literal, usa FAISS para encontrar variantes.
    #     4 Genera nuevas SQL reemplazando esos literales.
    #     5 Devuelve lista: [original, variante1, variante2, ...].
    #     """
    #     sql = self.generate_sql(question).strip()

    #     if sql.lower() == "select 'not available';":
    #         # No hace variantes si no hay query real
    #         return [sql]

    #     # 1) Encontrar todos los '... = 'valor'' o LIKE '%valor%'
    #     pattern = r"(=|LIKE)\s*['\"](.*?)['\"]"
    #     matches = re.findall(pattern, sql, flags=re.IGNORECASE)

    #     # Si no hay literales, devuelve solo la original
    #     if not matches:
    #         return [sql]

    #     # 2) Para cada literal, buscar variantes FAISS
    #     # Construir un mapa: literal -> [(tabla, columna, texto_relacionado)]
    #     replacements = {}

    #     for op, literal in matches:
    #         literal_clean = literal.strip()
    #         literal_clean = literal_clean.strip('%')
            
    #         # Consultar inverted_index: recupera lista de indices para este literal en todas las tablas/columnas
    #         # Aquí asumimos FAISS ya tiene el mapeo de (tabla, columna)
    #         faiss_hits = []
    #         for (table, column) in self.faiss_runtime.inverted_index.keys():
    #             hits = self.faiss_runtime.search(
    #                 word=literal_clean,
    #                 table=table,
    #                 column=column,
    #                 k=3,
    #                 threshold=0.60
    #             )
    #             print("------")
    #             print(literal_clean, (table, column))
    #             print(hits)
    #             print("------")
    #             faiss_hits.extend(hits)
    #         # Ordena por similitud descendente
    #         faiss_hits = sorted(faiss_hits, key=lambda x: -x[1])
    #         if faiss_hits:
    #             replacements[literal_clean] = [t[2][2] for t in faiss_hits]  # lista de textos relacionados

    #     # 3) Generar variantes reemplazando cada literal por cada alternativa posible
    #     variants = [sql]  # original primero

    #     for literal, alternatives in replacements.items():
    #         new_variants = []
    #         for variant_sql in variants:
    #             for alt in alternatives:
    #                 # reemplazo literal simple
    #                 variant_sql_alt = re.sub(
    #                     rf"(['\"]){literal}(['\"])", 
    #                     f"'{alt}'", 
    #                     variant_sql
    #                 )
    #                 new_variants.append(variant_sql_alt)
    #         variants = new_variants

    #     return variants



    # def generate_sql_with_alternatives(self, question: str, table: str, column: str) -> str:
    #     """
    #     Variante mejorada:
    #     1 Intenta generar la SQL directamente.
    #     2 Si falla o el resultado no es bueno, usa FAISS para buscar sinónimos o frases relacionadas.
    #     3 Reintenta con versiones alternativas de la pregunta.
    #     """
    #     base_sql = self.generate_sql(question)

    #     if base_sql.strip().lower() == "select 'not available';":
    #         # Buscar alternativas usando FAISS
    #         alternatives = self.faiss_runtime.search(
    #             word=question,
    #             table=table,
    #             column=column,
    #             k=3,
    #             threshold=0.60
    #         )

    #         for word, sim, (tabla, col, text) in alternatives:
    #             # Reintentar con texto relacionado
    #             alt_sql = self.generate_sql(text)
    #             if alt_sql.strip().lower() != "select 'not available';":
    #                 return alt_sql

    #         # Si ninguna alternativa da buen resultado, devolver fallback
    #         return base_sql

    #     return base_sql