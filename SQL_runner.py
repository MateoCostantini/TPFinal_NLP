import sqlite3
import json
from typing import Any, Dict, List


class SQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query)

                if query.strip().lower().startswith('select'):

                    rows = cursor.fetchall()
                    result = [dict(row) for row in rows]
                    resp =  {"sql": result, "status": "select" , "error":None}
                else:
                    conn.commit()
                    result = []
                    print("Operación realizada correctamente ")

                    resp = {"sql": result, "status": "modify" ,"error":None}


        except Exception as e:
            # Registro de error para producción
            print(f"Error executing query: {e}")
            resp = {"sql": [], "status": "error" , "error":e}
        
        finally:
            conn.close()
        return resp

    

    def execute_query_and_print(self, query: str):
        result = self.execute_query(query)
        if result["error"] != None:
            print(json.dumps(result["sql"], indent=2))
        else: 
            print("Hubo un error al ejecutar la query")

