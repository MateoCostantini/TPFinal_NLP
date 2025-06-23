from openai import AzureOpenAI
from typing import Any, Dict, List
import json



class TableAnswerer:
    def __init__(self,
                azure_api_key: str,
                azure_api_endpoint: str, 
                azure_api_version: str,
                deployment: str):
        
        azure_client = AzureOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_api_endpoint,
            api_version=azure_api_version
        )
        self.client = azure_client
        self.model = deployment

    def answer(self, question: str, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "I couldn't find any information to answer your question."

        table_preview = json.dumps(rows[:100], indent=2, ensure_ascii=False)
        system_prompt = (
            "You are a helpful assistant. You are given a user's question and the preview of the SQL result table."
            "A previous model has already preprocessed the user's message and built a table with all the information you need to respond. The table will be delivered to you."
            "The table will be provided for context. The structure will be clear, including column names and their values."
            "Answer the question in a clear, short, natural language answer, summarizing the table content. "
            "If possible, highlight key numbers, names, or insights."
            
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Table Preview (JSON):\n{table_preview}\n\n"
            "Answer:"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )

        return resp.choices[0].message.content.strip()

