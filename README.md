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

| Requisito | Detalle |
|---|---|
| Python 3.12 o superior | Verificar con `python --version` (en Windows también `py --version`) |
| Git | Para clonar el repositorio |
| Cuenta en [Groq](https://console.groq.com) | Plan gratuito; se usa su API key |
| Cuenta en [MongoDB Atlas](https://cloud.mongodb.com) | Plan gratuito (cluster *Free*) |
| Espacio en disco | ~2,5 GB (entorno virtual ~1,5 GB, modelo de embeddings ~470 MB, fuentes ~140 MB) |
| Conexión a internet | Para descargar las fuentes y usar Groq, Atlas, Open Library y Wikipedia |

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/velizabetho/agente-rag-isy0101.git
cd agente-rag-isy0101
```

Todos los comandos siguientes se ejecutan desde esta carpeta (la raíz del proyecto).

### 2. Crear y activar el entorno virtual

| Sistema | Crear | Activar |
|---|---|---|
| Windows (PowerShell) | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `python -m venv .venv` | `.venv\Scripts\activate.bat` |
| Linux / macOS | `python3 -m venv .venv` | `source .venv/bin/activate` |

Con el entorno activo, la terminal muestra `(.venv)` al inicio de la línea. Si PowerShell bloquea la activación, ejecutar una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Instala LangGraph, LangChain-Groq, PyMongo, sentence-transformers (con PyTorch), PyMuPDF, pandas y Streamlit. Toma algunos minutos la primera vez.

### 4. Obtener la API key de Groq

1. Entrar a [console.groq.com/keys](https://console.groq.com/keys) y crear una key (empieza con `gsk_`).
2. Guardarla para el paso 6.

### 5. Preparar MongoDB Atlas

1. Crear un cluster gratuito (*Free*) en [cloud.mongodb.com](https://cloud.mongodb.com). Cualquier región sirve; São Paulo es la más cercana a Chile.
2. *Database Access* → crear un usuario con contraseña y rol *Read and write to any database* (o `readWrite` sobre la base `clubes_lectura`).
3. *Network Access* → agregar `0.0.0.0/0` para poder conectarse desde cualquier red. El acceso queda protegido por el usuario y la contraseña.
4. *Connect* → *Drivers* → copiar el *connection string* (`mongodb+srv://...`) y reemplazar `<password>` por la contraseña del usuario.

No hace falta crear colecciones ni índices: el paso 7 los crea.

### 6. Configurar las variables de entorno

```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```

Abrir `.env` y completar al menos:

```ini
GROQ_API_KEY=gsk_...
MONGODB_URI=mongodb+srv://usuario:password@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

El resto de las variables ya trae valores por defecto (ver [Variables de entorno](#variables-de-entorno)). El archivo `.env` no se sube a git.

### 7. Preparar los datos

```bash
python -m bookclub.setup
```

Este comando hace cuatro cosas y se puede volver a ejecutar cuando se quiera partir de cero:

| Paso | Qué hace | Resultado esperado |
|---|---|---|
| 1/4 | Descarga las fuentes públicas del SNBP a `data/raw/` (manual, 16 catálogos, planilla de la Biblioteca Pública Digital; ~140 MB) | Los archivos ya descargados se omiten |
| 2/4 | Arma la base SQLite `data/bookclub.db` | `catalog_titles: 1806`, `ebooks: 3430`, `clubs: 195` |
| 3/4 | Divide el manual en fragmentos, calcula los embeddings y los carga en Atlas con su índice vectorial | `chunks: 93` |
| 4/4 | Crea el índice vectorial de la memoria de largo plazo | `done` |

La primera vez también descarga el modelo de embeddings `intfloat/multilingual-e5-small` (~470 MB). Todo el proceso toma unos minutos.

> Volver a ejecutar `bookclub.setup` (o `python -m bookclub.data.build_db`) reconstruye la base SQLite y **borra las reservas, planes y sesiones registradas**.

### 8. Ejecutar la aplicación

```bash
streamlit run app.py
```

Se abre `http://localhost:8501` en el navegador. Si no se abre solo, entrar a esa dirección.

1. En el panel izquierdo elegir la **región** y el **club** (por ejemplo *Metropolitana → Gente de palabra*).
2. Escribir en el chat, por ejemplo: *¿Cuánto debería durar un ciclo si nos juntamos cada semana?* o *Reserva 8 copias de Sub terra del 2026-11-02 al 2026-12-28*.
3. Bajo cada respuesta, **🔧 Trazabilidad** muestra el plan de tareas, las herramientas usadas y lo que devolvieron.
4. La pestaña **🗂️ Registro del club** muestra el plan activo, las reservas, las sesiones y la memoria del club.
5. **Nueva conversación** abre un thread nuevo (la memoria de largo plazo del club se mantiene).

Para detener la aplicación: `Ctrl + C` en la terminal.

## Solución de problemas

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `RuntimeError: MONGODB_URI is not set` | Falta el `.env` o la variable | Revisar el paso 6 y ejecutar desde la raíz del proyecto |
| `ServerSelectionTimeoutError` al conectar a Atlas | Cluster pausado, red no habilitada o contraseña incorrecta | En Atlas: reanudar el cluster (se pausa tras 30 días sin uso), revisar *Network Access* y la contraseña en `MONGODB_URI` |
| `Error code: 429 ... tokens per minute (TPM)` | Límite de tokens por minuto del plan gratuito de Groq | Esperar unos segundos; el cliente reintenta solo |
| `Error code: 429 ... tokens per day (TPD)` | Se agotó la cuota diaria (200.000 tokens por modelo) | Esperar a que se renueve o activar el nivel *Dev* de Groq |
| La primera respuesta tarda | Se carga el modelo de embeddings en memoria | Solo ocurre una vez por ejecución de la app |
| `Port 8501 is already in use` | Otra app de Streamlit abierta | Cerrarla o usar `streamlit run app.py --server.port 8502` |
| Aviso `unauthenticated requests to the HF Hub` | Descarga del modelo sin token de Hugging Face | Se puede ignorar |

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
