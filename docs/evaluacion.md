# Evaluación

## Método

- **Dataset:** [`evaluation/dataset.json`](../evaluation/dataset.json), 33 preguntas con respuesta verificada en las fuentes.

| Categoría | Preguntas | Qué prueba |
|---|---|---|
| `rag` | 10 | Respuestas del manual de orientaciones; cada una indica las páginas donde está la respuesta |
| `sql` | 8 | Datos de catálogos, Biblioteca Pública Digital y clubes |
| `externa` | 2 | Open Library y Wikipedia |
| `accion` | 7 | Reservar, planificar, registrar sesión, reprogramar, rechazar un título de otra región, guardar memoria |
| `no_tool` | 3 | Saludos y temas ajenos: no debe usar herramientas |
| `falso_positivo` | 3 | Parecen requerir datos pero no: cálculos con datos del usuario, conceptos generales, información que el sistema no tiene |

- **Métricas:**
  - *Tool accuracy*: se llamaron las herramientas esperadas (o ninguna, en `no_tool` y `falso_positivo`).
  - *Answer accuracy*: un LLM juez (`gpt-oss-20b`) compara la respuesta con la esperada y devuelve `{correct, reason}`.
  - *Retrieval hit@5*: en preguntas `rag`, al menos uno de los 5 fragmentos recuperados es de una página esperada.
- Cada pregunta corre en un thread nuevo. Las acciones se ejecutan sobre una copia temporal de la base SQLite y en orden (reservar → planificar → registrar → reprogramar).
- Ejecución: `python -m evaluation.evaluate` (guarda el avance después de cada pregunta; `--resume` continúa una corrida interrumpida).

## Corrida 1 — prompt v1 (parcial)

La corrida se detuvo en la pregunta 25 por el límite diario de tokens del plan gratuito de Groq (200.000 tokens por día para `gpt-oss-120b`). Resultado de las 24 preguntas evaluadas:

| Categoría | Preguntas | Tool accuracy | Answer accuracy |
|---|---|---|---|
| rag | 10 | 8/10 | 6/10 |
| sql | 8 | 8/8 | 8/8 |
| externa | 2 | 2/2 | 2/2 |
| accion | 4 | 4/4 | 3/4 |
| **Total** | **24** | **22/24 (92 %)** | **19/24 (79 %)** |

### Análisis de errores

| Pregunta | Error | Origen | Cambio |
|---|---|---|---|
| q08 (personas y redes regionales del programa) | Consultó la base de datos en vez del manual | **Decisión de herramienta**: la base solo tiene los 195 clubes con ficha, no las cifras oficiales del programa (más de 270 clubes, 2700 personas, 16 redes) | Mismo cambio del prompt v2 |
| q08 (personas y redes regionales del programa) | Consultó la tabla de clubes en vez del manual | **Decisión de herramienta**: contó clubes de la base (195) en vez de la cifra oficial del manual (más de 270) | Mismo cambio del prompt v2 |
| q09 (límite de Zoom gratuito) | El fragmento correcto (p. 55) se recuperó, pero el juez marcó incorrecta la respuesta | **Respuesta esperada**: exigía un segundo dato no esencial (que Zoom requiere instalación) | Se dejó solo el dato esencial (40 minutos) |
| q10 (nombre del curso y número de módulos) | Respuesta incorrecta con la herramienta correcta | **Recuperación**: los módulos están repartidos en 3 fragmentos (pp. 57-59) y el número total no aparece escrito | Se mantiene como pregunta difícil |
| q21 (reservar 8 copias) | Reservó bien, pero el juez marcó incorrecta la respuesta | **Respuesta esperada**: exigía informar las copias restantes, dato no pedido | Se dejó solo lo esencial (copias y fechas) |

## Corrida 2 — prompt v2

Pendiente: se ejecuta cuando se renueve la cuota diaria de Groq (`python -m evaluation.evaluate`). El resultado queda en `evaluation/results/` y en la pestaña **Evaluación** de la app.
