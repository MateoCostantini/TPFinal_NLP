import os
import json
import argparse
import logging

from preprocessor_database import SQLitePreprocessor
from faiss_runtime import SQLiteFAISSRuntime
from gen_SQL import NLToSQLService
from SQL_runner import SQLExecutor
from table_to_NL import TableAnswerer
from classifier import ClassifierService
from modifier_SQL import SQLModifierModel
from evaluator import Evaluator

#================= MUY BUENO ESTO PARA CORRERLO POR COMANDO:
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Inspección de esquema de base SQLite")
#     parser.add_argument("--db", required=True, help="Ruta al archivo .db de entrada")
#     args = parser.parse_args()


class Tablon:
    def __init__(
        self,
        db_path: str,
        azure_api_key_embeddings: str,
        azure_api_endpoint_embeddings: str,
        azure_api_version_embeddings: str,
        deployment_embeddings: str,
        azure_api_key_model: str,
        azure_api_endpoint_model: str,
        azure_api_version_model: str,
        deployment_model: str,
        preprocess = False

    ):
        #self.conversation_history = []  
        self.db_path = db_path
        self.DB_folder = os.path.dirname(db_path)
        self.azure_api_key_embeddings = azure_api_key_embeddings
        self.azure_api_endpoint_embeddings = azure_api_endpoint_embeddings
        self.azure_api_version_embeddings = azure_api_version_embeddings
        self.deployment_embeddings=deployment_embeddings
        self.azure_api_key_model = azure_api_key_model
        self.azure_api_endpoint_model =azure_api_endpoint_model
        self.azure_api_version_model = azure_api_version_model
        self.deployment_model = deployment_model


        self.preprocessor = SQLitePreprocessor(
            sqlite_path=self.db_path,
            azure_api_key=self.azure_api_key_embeddings,
            azure_api_base=self.azure_api_endpoint_embeddings,
            azure_api_version=self.azure_api_version_embeddings,
            embedding_deployment=self.deployment_embeddings
        )   

        self.preprocessor.create_log_file()

        if preprocess:
            self.preprocessor.run()  
            self.preprocessor.create_scheme()
        
        # ===  Consultar (Busca reemplazos mas cercanos (RAG)) ===
        self.runtime_FAISS = SQLiteFAISSRuntime(
            DB_folder = self.DB_folder,
            azure_api_key=self.azure_api_key_embeddings,
            azure_api_base=self.azure_api_endpoint_embeddings,
            azure_api_version=self.azure_api_version_embeddings,
            embedding_deployment=self.deployment_embeddings
        )

        # ================= Modelo Clasificador ========================
        self.classifier_service = ClassifierService(
            db_path=self.db_path,
            azure_api_key=self.azure_api_key_model,
            azure_api_endpoint=self.azure_api_endpoint_model,
            azure_api_version=self.azure_api_version_model,
            deployment=self.deployment_model
        )

        # ================= Modelo GEN SQL =============================
        self.gen_SQL_service = NLToSQLService(
            db_path=self.db_path,
            azure_api_key=self.azure_api_key_model,
            azure_api_endpoint=self.azure_api_endpoint_model,
            azure_api_version=self.azure_api_version_model,
            deployment=self.deployment_model,
            faiss_runtime=self.runtime_FAISS
        )

        # ================= Modelo Modificador SQL =====================
        self.modifier_SQL_service = SQLModifierModel(
            db_path=self.db_path,
            azure_api_key=self.azure_api_key_model,
            azure_api_endpoint=self.azure_api_endpoint_model,
            azure_api_version=self.azure_api_version_model,
            deployment=self.deployment_model
        )

        # =================  SQL executor =============================
        self.executor = SQLExecutor(
            db_path = self.db_path
        )

        # ================ Answer back to user =====================
        self.answerer = TableAnswerer(
        azure_api_key=self.azure_api_key_model,
        azure_api_endpoint=self.azure_api_endpoint_model,
        azure_api_version=self.azure_api_version_model, 
        deployment=self.deployment_model
        )

        # ================ Evaluación del modelo =====================
        self.evaluator = Evaluator(
            db_path="database\sakila_database\sakila_master.db",
            azure_api_key=self.azure_api_key_model,
            azure_api_endpoint=self.azure_api_endpoint_model,
            azure_api_version=self.azure_api_version_model,
            deployment=self.deployment_model,
            faiss_runtime=self.runtime_FAISS
        )



    def preprocess_dataset(self):
        # Hay que correrlo solo una vez (Ya corrido)
        self.preprocessor.run()  
        self.preprocessor.create_scheme()  

    def answer(self, question):

        # 1. Clasificar la consulta
        category = self.classifier_service.classify_query(question)

        if category == "observes_database":
            sql = self.gen_SQL_service.generate_sql_with_embedded_variants(question)

            print("Final sql: ", sql)

            # if sql.lower() == "select 'not available';":
            #     print("Sorry, I couldn't generate a valid SQL query for your question.")

            execution = self.executor.execute_query(sql)

            tabla = execution['sql']
            print(tabla)

            resp = self.answerer.answer(question, sql, tabla)
            print("Answer: ", resp)


        elif category == "modifies_database":
            sql, undo_sql = self.modifier_SQL_service.generate_sql(question)
            print("sql: ", sql)
            print("undo_sql: ", undo_sql)
            execution = self.executor.execute_query(sql)
            print("result: ",execution)
            if execution["error"] == None:
                self.preprocessor.log_db_action(sql, undo_sql)

    def evaluate(self):       
        resultados = self.evaluator.evaluar_modelo_sql(
            modelo_generador=self.gen_SQL_service)
        
        # Imprime o devuelve los resultados de forma resumida o detallada
        avg_sim = sum(r["similitud"] for r in resultados) / len(resultados)
        print(f"Similitud promedio en iteración : {avg_sim:.2f}")





##### Ver caso de que la tabla sea excesivamente larga y haya qeu recortarla.
##### ver que modelo es mejor para cada una de las partes.
##### ver si el preprocesamiento lo puede hacer en gpu.
#### dirigir el pipeline segun corresponda. (not available, status: error, tabla vacia, etc.)
##### en vez de que el que elige la mejor query reescriba la query podria capaz elegir el indice de la mejor y despues la agarro [indice]




# tablon = Tablon(
#     db_path= "database\sakila_database\sakila_master.db", #args.db , (si hace evaluacion tiene que ser database\sakila_database\sakila_master.db sino args.db)
#     azure_api_key_embeddings="CmzNoGQRpdysAdiAVniDBG7PDJvup7GvjWdolgOpdLe6FNotvVWMJQQJ99BFACYeBjFXJ3w3AAABACOGUZRT",
#     azure_api_endpoint_embeddings="https://mateo-openai.openai.azure.com/",
#     azure_api_version_embeddings="2023-12-01-preview",
#     deployment_embeddings="text-embedding-3-small-tablon",
#     azure_api_key_model="5gENmcyHfZ6TrGM6ictkRXzD8IL75KijInGCjGbq7IqgEilgdpeyJQQJ99BFACfhMk5XJ3w3AAAAACOGVBGv",
#     azure_api_endpoint_model="https://mateo-mbio42gi-swedencentral.cognitiveservices.azure.com/",
#     azure_api_version_model="2023-12-01-preview",
#     deployment_model="gpt-4o-mini-tute",
#     preprocess = False ## conseguir de args.p
#     )




# while True:
#     print("")
#     question = input("-> ")
#     if question.lower() in ["q", "quit"]:
#         break
#     elif question.lower() in ["eval", "evaluate"]:
#         n = int(input("How many iterations do you want to evaluate? "))
#         print("")
#         tablon.evaluate(n)
#     else:
#         print("")
#         tablon.answer(question)
#     print("")



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Inspección de esquema de base SQLite")
#     parser.add_argument("--db", required=True, help="Ruta al archivo .db de entrada")
#     args = parser.parse_args()

def main():
    parser = argparse.ArgumentParser(description="Inspección de esquema de base SQLite y evaluación del modelo.")
    parser.add_argument("--db", help="Ruta al archivo .db de entrada (obligatorio si no se usa --eval)")
    parser.add_argument("--preprocess", action="store_true", help="Si se incluye, realiza preprocesamiento del esquema")
    parser.add_argument("--eval", action="store_true", help="Evalúa el modelo con el dataset predefinido")

    args = parser.parse_args()

    if args.eval:
        print(" Ejecutando evaluación...")
        tablon = Tablon(
            db_path="database/sakila_database/sakila_master.db",  # ruta fija para evaluación
            azure_api_key_embeddings="CmzNoGQRpdysAdiAVniDBG7PDJvup7GvjWdolgOpdLe6FNotvVWMJQQJ99BFACYeBjFXJ3w3AAABACOGUZRT",
            azure_api_endpoint_embeddings="https://mateo-openai.openai.azure.com/",
            azure_api_version_embeddings="2023-12-01-preview",
            deployment_embeddings="text-embedding-3-small-tablon",
            azure_api_key_model="5gENmcyHfZ6TrGM6ictkRXzD8IL75KijInGCjGbq7IqgEilgdpeyJQQJ99BFACfhMk5XJ3w3AAAAACOGVBGv",
            azure_api_endpoint_model="https://mateo-mbio42gi-swedencentral.cognitiveservices.azure.com/",
            azure_api_version_model="2023-12-01-preview",
            deployment_model="gpt-4o-mini-tute",
            preprocess=False  # no se requiere preprocesamiento en evaluación
        )

        tablon.evaluate()

    else:
        if not args.db:
            parser.error("El argumento --db es obligatorio si no se usa --eval")

        print(f" Ejecutando sistema en modo interactivo con base: {args.db}")
        tablon = Tablon(
            db_path=args.db,
            azure_api_key_embeddings="CmzNoGQRpdysAdiAVniDBG7PDJvup7GvjWdolgOpdLe6FNotvVWMJQQJ99BFACYeBjFXJ3w3AAABACOGUZRT",
            azure_api_endpoint_embeddings="https://mateo-openai.openai.azure.com/",
            azure_api_version_embeddings="2023-12-01-preview",
            deployment_embeddings="text-embedding-3-small-tablon",
            azure_api_key_model="5gENmcyHfZ6TrGM6ictkRXzD8IL75KijInGCjGbq7IqgEilgdpeyJQQJ99BFACfhMk5XJ3w3AAAAACOGVBGv",
            azure_api_endpoint_model="https://mateo-mbio42gi-swedencentral.cognitiveservices.azure.com/",
            azure_api_version_model="2023-12-01-preview",
            deployment_model="gpt-4o-mini-tute",
            preprocess=args.preprocess
        )

        while True:
            print("")
            question = input("-> ")
            if question.lower() in ["q", "quit"]:
                break
            elif question.lower() in ["eval", "evaluate"]:
                print("")
                tablon.evaluate()
            else:
                print("")
                tablon.answer(question)
            print("")


if __name__ == "__main__":
    main()