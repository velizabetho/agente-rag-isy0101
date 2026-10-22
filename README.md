# agente-rag-isy0101

Asistente para personas que conducen **clubes de lectura de las bibliotecas públicas de Chile** (Sistema Nacional de Bibliotecas Públicas, SNBP). Proyecto semestral de **Ingeniería de Soluciones con Inteligencia Artificial (ISY0101)** — DuocUC, 2026-2.

El caso está descrito en [`docs/caso.md`](docs/caso.md) y la arquitectura en [`docs/arquitectura.md`](docs/arquitectura.md).

| Etapa | Evaluación | Qué incluye el proyecto |
|---|---|---|
| 1 | Diseño de Solución con LLM y RAG | RAG sobre el manual oficial del SNBP (fuente interna) + consultas SQL a los catálogos + Open Library y Wikipedia (fuentes externas) |
| 2 | Desarrollo de un Agente Funcional | Agente LangGraph con herramientas de consulta, escritura y razonamiento, memoria de corto y largo plazo y planificación de tareas |

## Qué hace

Quien conduce un club elige su club en la interfaz y conversa con el agente, que puede:

- **Responder dudas de metodología** con el manual *Orientaciones para mediar y gestionar clubes de lectura* (SNBP, 2025), citando la página.
- **Buscar libros** en los 16 catálogos regionales para clubes (1.806 títulos) y en la Biblioteca Pública Digital (3.430 títulos).
- **Verificar copias disponibles** en ciertas fechas y **reservarlas** (solo del catálogo de la región del club).
- **Armar el calendario de lectura** según la frecuencia del club, **registrar sesiones**, detectar si el grupo va atrasado y **reprogramar**.
- **Preparar sesiones** con información externa del libro (Open Library) y del autor (Wikipedia) y guardar la guía.
- **Recordar** preferencias y acuerdos del club entre conversaciones.

## Estructura

```
bookclub/
  config.py            variables de entorno (.env)
  data/                descarga de fuentes, parser de catálogos, clubes, armado de SQLite
  rag/                 ingesta del manual a MongoDB Atlas y búsqueda vectorial
  agent/               herramientas, prompts y grafo LangGraph
  records.py           reglas de reservas, planes y registros de sesiones
  planning.py          cálculo del calendario de lectura
  sql.py               consultas SQL de solo lectura (text-to-SQL)
  memory.py            memoria de largo plazo por club
  external.py          Open Library y Wikipedia
  setup.py             prepara todo con un comando
app.py                 interfaz Streamlit (chat, registro del club, evaluación)
evaluation/            dataset de 33 preguntas, script y resultados
tests/                 pruebas automáticas (pytest)
data/                  listado de clubes (CSV); el resto se genera
```

## Requisitos

- Python 3.12 o superior.
- Una API key de [Groq](https://console.groq.com/keys) (plan gratuito).
- Un cluster de [MongoDB Atlas](https://cloud.mongodb.com) (plan gratuito) con un usuario de lectura/escritura. En *Network Access* se habilita `0.0.0.0/0` para poder conectarse desde cualquier red; el acceso queda protegido por el usuario y la contraseña de `MONGODB_URI`.

## Instalación y ejecución

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar GROQ_API_KEY y MONGODB_URI

python -m bookclub.setup    # descarga fuentes, arma SQLite, ingesta el manual en Atlas
streamlit run app.py        # abre http://localhost:8501
```

`bookclub.setup` se puede volver a ejecutar cuando cambien las fuentes: reconstruye la base SQLite y la colección del manual desde cero. La primera vez descarga el modelo de embeddings (`intfloat/multilingual-e5-small`, ~470 MB).

## Pruebas y evaluación

```bash
pytest -q                          # pruebas sin LLM ni red (reglas, SQL seguro, grafo con modelo simulado)
python -m evaluation.evaluate      # evaluación completa con el LLM real
python -m evaluation.evaluate q01 q11   # solo algunas preguntas
```

La evaluación mide:

- **Tool accuracy:** el agente usó las herramientas esperadas (o ninguna, en saludos y falsos positivos).
- **Answer accuracy:** un LLM juez compara la respuesta con la esperada.
- **Retrieval hit@5:** en preguntas del manual, si se recuperó un fragmento de la página correcta.

Las acciones de escritura se evalúan sobre una copia temporal de la base, sin tocar los registros reales. Los resultados quedan en `evaluation/results/run_<fecha>.json` y se ven en la pestaña **Evaluación** de la app.

> El plan gratuito de Groq limita los tokens por minuto de cada modelo, por eso la evaluación completa toma varios minutos.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `GROQ_API_KEY` | API key de Groq |
| `LLM_MODEL` | Modelo del agente (`openai/gpt-oss-120b`) |
| `PLANNER_MODEL` / `JUDGE_MODEL` | Modelo del planificador y del juez de evaluación (`openai/gpt-oss-20b`) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `local` + `intfloat/multilingual-e5-small`, u `openai` + `text-embedding-3-small` |
| `OPENAI_API_KEY` | Solo si `EMBEDDING_PROVIDER=openai` |
| `MONGODB_URI` / `MONGODB_DB` | Conexión a Atlas y nombre de la base |
| `LANGSMITH_*` | Trazas en LangSmith (opcional) |

## Fuentes de datos

- Sistema Nacional de Bibliotecas Públicas. (2025). *Orientaciones para mediar y gestionar clubes de lectura*.
- Sistema Nacional de Bibliotecas Públicas. (s.f.). *Catálogos regionales para clubes de lectura* y listado de la Biblioteca Pública Digital.
- Sistema Nacional de Bibliotecas Públicas. (s.f.). Fichas de los clubes de lectura de bibliotecas públicas (`data/clubes_lectura_snbp_2026.csv`).
- Open Library API y Wikipedia en español (fuentes externas, en tiempo real).

## Ramas

- `main`: versión estable, la que se entrega.
- `dev`: desarrollo. Los cambios pasan a `main` cuando están probados.
