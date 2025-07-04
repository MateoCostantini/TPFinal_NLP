import json
from typing import Dict, List, Any
from openai import AzureOpenAI
import instructor
import os
from pydantic import BaseModel, Field
from sqlglot import exp



class SQLResponse(BaseModel):
    reasoning: str = Field(
        description="Clearly think through what changes need to be made to the database based on the user's message. Justify why these modifications are necessary, which tables are used, and which columns are affected."
    )
    SQL: str = Field(
        description="One or more correct SQL statements to apply the modifications to the database. May include INSERT, UPDATE, or DELETE."
    )
    SQL_undo: str = Field(
        description="One or more correct SQL statements to exactly undo the modifications applied by the SQL field. Think about which records were affected, and generate the appropriate inverse commands (DELETE if you previously did INSERT, INSERT if you previously did DELETE, UPDATE with previous values if you did UPDATE). Use reasonable default values if not everything can be inferred."
    )

class SQLSelected(BaseModel):
    SQL: str = Field(..., description="Choose the SQL Query that best fits the user's request.")


class SQLModifierModel:
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
            "You are given a user's request to modify a database, and a list of different SQL queries that could fulfill the request."
            "Your task is to carefully analyze which SQL query is the most correct, complete, and precise for doing what's requested."
            "Return ONLY the chosen SQL query as plain text, with NO explanations, NO markdown, and NO extra text.\n\n"
            f"Request: {question}\n\n"
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
    


    def generate_sql(self, question: str) -> str:
        schema_repr = json.dumps(self.schema, indent=2, ensure_ascii=False)
        prompt = (
            "You are an expert in SQL specialized in modifying databases. "
            "Your job is to generate SQL statements that faithfully reflect the changes requested by the user. "
            "The user will tell you in natural language what happened (for example, add X element to Y table). "
            "Based on the provided database schema, generate a sequence of SQL statements that modify the database accordingly. "
            "IMPORTANT:\n"
            "- Only respond with SQL code (no explanations, no markdown).\n"
            "- The statements must be syntactically correct.\n"
            "- If something cannot be inferred, use reasonable default values.\n"
            "- Make sure to respect the columns defined in each table.\n"
            "- Do not generate SELECTs. Only INSERT, UPDATE, or DELETE.\n"
            "- Fill in NOT NULL fields with default values if no values are specified. But if they allow it, use NULLs."
            "- Always include all required columns for an insertion.\n\n"
            "- Additionally, for every change made, also generate the SQL code that completely undoes it. "
            "This code must revert all the effects caused by the modification code. "
            "If you did an INSERT, the undo is a DELETE. If you did an UPDATE, the undo is an UPDATE with the previous values. "
            "If you did a DELETE, the undo is an INSERT with the original values. "
            "For numeric primary keys when no value is specified, generate a random 10-digit number that looks chaotic, such as 9372018456 or 5829301745. Avoid always using the same number like 1234567890."
            "Use reasonable default values when the exact previous state cannot be known.\n\n"
            "\nBelow I will provide you with the tables and their relationships:\n\n" + schema_repr
        )

        user_question = (
            f"Request: {question}\n\n"
            "Return only the corresponding SQL query (without explanations)."
        )


        resp = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question}
            ],
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            response_model=SQLResponse,
        )
        sql = resp.SQL
        sql_undo = resp.SQL_undo
        return sql, sql_undo