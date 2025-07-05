import os
import argparse

from preprocessor_database import SQLitePreprocessor
from faiss_runtime import SQLiteFAISSRuntime
from gen_SQL import NLToSQLService
from SQL_runner import SQLExecutor
from table_to_NL import TableAnswerer
from classifier import ClassifierService
from modifier_SQL import SQLModifierModel
from evaluator import Evaluator



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
        self.conversation_history = []  
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
        
        # ===  Consultar (Busca reemplazos mas cercanos) ===
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
            db_path=self.db_path,
            azure_api_key=self.azure_api_key_model,
            azure_api_endpoint=self.azure_api_endpoint_model,
            azure_api_version=self.azure_api_version_model,
            deployment=self.deployment_model,
            faiss_runtime=self.runtime_FAISS
        )



    def preprocess_dataset(self):
        # Hay que correrlo solo una vez por dataset
        self.preprocessor.run()  
        self.preprocessor.create_scheme()  

    


    def answer_with_retries(self, question: str, max_retries: int = 3):
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history = self.conversation_history[-12:]

        category = self.classifier_service.classify_query(question)

        if category == "modifies_database":
            sql, undo_sql = self.modifier_SQL_service.generate_sql(question)

            self.conversation_history.append({"role": "assistant", "content": sql})
            self.conversation_history = self.conversation_history[-12:]

            print("sql: ", sql)
            execution = self.executor.execute_query(sql)
            if execution["error"] == None:
                self.preprocessor.log_db_action(sql, undo_sql)
            
            self.conversation_history.append({"role": "assistant", "content": f'{execution["status"]}: error {execution["error"]}'})
            self.conversation_history = self.conversation_history[-12:]

        if category == "observes_database":
            attempt = 0
            while attempt < max_retries:
                attempt += 1

                sql = self.gen_SQL_service.generate_sql_with_embedded_variants(
                    question=question,
                    conversation=self.conversation_history[-12:]
                )

                self.conversation_history.append({"role": "assistant", "content": sql})
                self.conversation_history = self.conversation_history[-12:]

                execution = self.executor.execute_query(sql)
                rows = execution["sql"]

                if execution["error"] is not None:
                    flag = False
                    break
                if len(rows) > 0 and not ("'Not available'" in rows[0]):
                    flag = True
                    break
                else:
                    self.conversation_history.append({
                        "role": "user",
                        "content": "La última consulta SQL no devolvió resultados. Por favor, genera una consulta diferente que pueda recuperar datos."
                    })
                    self.conversation_history = self.conversation_history[-12:]
                    flag = False
            if not flag:
                sql = "SELECT no disponible;"
            resp = self.answerer.answer(question, sql, rows)
            print("Respuesta:", resp)
            self.conversation_history.append({"role": "assistant", "content": resp})
            self.conversation_history = self.conversation_history[-12:]


    def evaluate(self):       
        resultados = self.evaluator.evaluar_modelo_sql(
            modelo_generador=self.gen_SQL_service)
        
        avg_sim = sum(r["similitud"] for r in resultados) / len(resultados)
        print(f"Similitud promedio en iteración : {avg_sim:.2f}")

        self.evaluator.mostrar_graficos(resultados)



def main():
    parser = argparse.ArgumentParser(description="Inspección de esquema de base SQLite y evaluación del modelo.")
    parser.add_argument("--db", help="Ruta al archivo .db de entrada (obligatorio si no se usa --eval)")
    parser.add_argument("--preprocess", action="store_true", help="Si se incluye, realiza preprocesamiento del esquema")
    parser.add_argument("--eval", action="store_true", help="Evalúa el modelo con el dataset predefinido")

    args = parser.parse_args()

    if args.eval:
        print(" Ejecutando evaluación...")
        tablon = Tablon(
            db_path="database\evaluator\sakila_master.db",  # ruta fija para evaluación
            azure_api_key_embeddings="...",
            azure_api_endpoint_embeddings="...",
            azure_api_version_embeddings="...",
            deployment_embeddings="...",
            azure_api_key_model="...",
            azure_api_endpoint_model="...",
            azure_api_version_model="...",
            deployment_model="...",
            preprocess=False  # no se requiere preprocesamiento en evaluación
        )
        tablon.evaluate()

    else:
        if not args.db:
            parser.error("El argumento --db es obligatorio si no se usa --eval")

        print(f" Ejecutando sistema en modo interactivo con base: {args.db}")
        tablon = Tablon(
            db_path=args.db,
            azure_api_key_embeddings="...",
            azure_api_endpoint_embeddings="...",
            azure_api_version_embeddings="...",
            deployment_embeddings="...",
            azure_api_key_model="...",
            azure_api_endpoint_model="...",
            azure_api_version_model="...",
            deployment_model="...",
            preprocess=args.preprocess
        )

        while True:
            print("")
            question = input("-> ")
            if question.lower() in ["q", "quit"]:
                break
            else:
                print("")
                tablon.answer_with_retries(question)
            print("")


if __name__ == "__main__":
    main()