import os
import json
import argparse
from preprocessor_database import SQLitePreprocessor
from faiss_runtime import SQLiteFAISSRuntime
from gen_SQL import NLToSQLService
from SQL_runner import SQLExecutor
from table_to_NL import TableAnswerer

#================= MUY BUENO ESTO PARA CORRERLO POR COMANDO:
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Inspección de esquema de base SQLite")
#     parser.add_argument("--db", required=True, help="Ruta al archivo .db de entrada")
#     args = parser.parse_args()




# ===  Preprocesar ===
preprocessor = SQLitePreprocessor(
    sqlite_path="database/sakila-sqlite3/sakila_master.db", #args.db
    azure_api_key="CmzNoGQRpdysAdiAVniDBG7PDJvup7GvjWdolgOpdLe6FNotvVWMJQQJ99BFACYeBjFXJ3w3AAABACOGUZRT",
    azure_api_base="https://mateo-openai.openai.azure.com/",
    azure_api_version="2023-12-01-preview",
    embedding_deployment="text-embedding-3-small-tablon"
)

def preprocess():
    # Hay que correrlo solo una vez (Ya corrido)
    preprocessor.run()  
    preprocessor.create_scheme()  

#preprocess()


# ===  Consultar (Busca reemplazos mas cercanos (RAG)) ===
runtime = SQLiteFAISSRuntime(
    DB_folder = os.path.dirname("database/sakila-sqlite3/sakila_master.db"),
    azure_api_key="CmzNoGQRpdysAdiAVniDBG7PDJvup7GvjWdolgOpdLe6FNotvVWMJQQJ99BFACYeBjFXJ3w3AAABACOGUZRT",
    azure_api_base="https://mateo-openai.openai.azure.com/",
    azure_api_version="2023-12-01-preview",
    embedding_deployment="text-embedding-3-small-tablon"
)

# ================= Modelo GEN SQL =============================

service = NLToSQLService(
        db_path="database/sakila-sqlite3/sakila_master.db", #args.db
        azure_api_key="5gENmcyHfZ6TrGM6ictkRXzD8IL75KijInGCjGbq7IqgEilgdpeyJQQJ99BFACfhMk5XJ3w3AAAAACOGVBGv",
        azure_api_endpoint="https://mateo-mbio42gi-swedencentral.cognitiveservices.azure.com/",
        azure_api_version="2023-12-01-preview",
        deployment="gpt-4o-mini-tute",
        faiss_runtime=runtime
    )

# =================  SQL executor =============================

executor = SQLExecutor(
    db_path = "database/sakila-sqlite3/sakila_master.db" #args.db
    )


# ================ Answer back to user =====================
answerer = TableAnswerer(
        azure_api_key="5gENmcyHfZ6TrGM6ictkRXzD8IL75KijInGCjGbq7IqgEilgdpeyJQQJ99BFACfhMk5XJ3w3AAAAACOGVBGv",
        azure_api_endpoint="https://mateo-mbio42gi-swedencentral.cognitiveservices.azure.com/",
        azure_api_version="2023-12-01-preview", 
        deployment="gpt-4o-mini-tute")





while True:
    #question = "give me the name, surname and andress that have this email: Mike.Hillyer@sakilastaff.com"
    # quersion2 = "how many horror movies are there?"
    # 
    question = input("-> ")
    if question.lower() == "q":
        break
    print("")

    sql = service.generate_sql_with_embedded_variants(question)
    print("Final sql: ", sql)
    

    # if sql.lower() == "select 'not available';":
    #     print("Sorry, I couldn't generate a valid SQL query for your question.")


    execution = executor.execute_query(sql)
    tabla = execution['sql']
    print(tabla)

    resp = answerer.answer(question, tabla)
    print("Answer: ", resp)


##### Ver caso de que la tabla sea excesivamente larga y haya qeu recortarla.
##### ver que modelo es mejor para cada una de las partes.
##### ver si el preprocesamiento lo puede hacer en gpu.
#### dirigir el pipeline segun corresponda. (not available, status: error, tabla vacia, etc.)
##### en vez de que el que elige la mejor query reescriba la query podria capaz elegir el indice de la mejor y despues la agarro [indice]

