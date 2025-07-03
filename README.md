## Tablón

Tablón es un sistema interactivo de pregunta-respuesta que transforma lenguaje natural en comandos SQL, usando modelos de clasificación, generación y modificación basados en Azure OpenAI. Permite consultar y modificar bases de datos SQLite usando texto en lenguaje natural, con un enfoque RAG (Retrieval-Augmented Generation) y evaluación automática.

---

## ¿Qué hace?

1. Clasifica la intención de la consulta (lectura o modificación).
2. Genera SQL usando contexto y embeddings (RAG).
3. Ejecuta los comandos sobre una base SQLite.
4. Devuelve respuestas explicativas basadas en tablas.
5. Permite evaluar la performance de generación.

---

## Instalación

1. Clonar el repositorio
  git clone https://github.com/MateoCostantini/TPFinal_NLP.git
  cd TPFinal_NLP

2. Instalar dependencias
  pip install -r requirements.txt



## Organización del archivo .db
database/<nombre_de_tu_carpeta>/<tu_archivo>.db


## Importante: Preprocesamiento obligatorio
Antes de usar el sistema por primera vez, debes correr el preprocesamiento para:
* Generar los embeddings del esquema de tu base
* Crear el esquema en JSON utilizado internamente
Para eso, se debe ejecutar:
  python main.py --db database/<nombre_de_tu_carpeta>/<tu_archivo>.db --preprocess


## Uso interactivo del modelo
  python main.py --db database/<nombre_de_tu_carpeta>/<tu_archivo>.db


## Evaluacion del modelo
  python main.py --eval


