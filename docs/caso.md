# Caso: asistente para conductores de clubes de lectura

## Organización

**Sistema Nacional de Bibliotecas Públicas (SNBP)**, Servicio Nacional del Patrimonio Cultural, Chile.
Programa: **Redes Regionales de Clubes de Lectura**, a cargo de la Coordinación de Fomento Lector.

- Alrededor de 2700 personas participan cada año en más de 270 clubes de lectura de bibliotecas públicas, articulados en 16 redes regionales (SNBP, 2025, p. 9).
- Cada club lo conduce una persona voluntaria: funcionaria de la biblioteca o integrante de la comunidad (SNBP, 2025, p. 12).
- Cada región tiene un **catálogo regional** de libros físicos para clubes, con al menos 11 copias por título, que se reservan, prestan y devuelven para que circulen entre los clubes de la región (SNBP, 2025, pp. 18-19).
- La Biblioteca Pública Digital (bpdigital.cl) complementa con libros electrónicos; los títulos con licencia se reservan por correo (SNBP, s.f.).

## Problema

La persona que conduce un club, que es voluntaria, concentra dos tipos de trabajo (SNBP, 2025, pp. 22-24):

- **Administrativo y logístico:** inscripciones y asistencia, calendario de sesiones, reserva del espacio, préstamo y devolución de libros del catálogo regional, registro de sesiones y acuerdos.
- **Mediación lectora:** presentar la obra, preparar preguntas abiertas, conexiones con otros libros y autores, dinámicas de inicio y cierre.

La información para hacerlo está repartida en un manual PDF de más de 70 páginas, en 16 catálogos regionales en PDF y en una planilla de la Biblioteca Pública Digital. El seguimiento de cada club se lleva a mano.

## Objetivo general

Reducir el tiempo que dedica quien conduce un club a planificar y preparar sus sesiones, con un agente que consulta el catálogo y las orientaciones oficiales, propone y prepara sesiones, y lleva el registro del avance del club.

Los objetivos medibles (tiempos, métricas de evaluación) se definen en la propuesta.

## Datos disponibles

| Fuente | Tipo | Uso | Origen |
|---|---|---|---|
| *Orientaciones para mediar y gestionar clubes de lectura* (SNBP, nov. 2025), 70+ págs. | PDF, texto | RAG: metodología, estructura de sesiones, recomendaciones | Interna (SNBP), pública |
| Catálogos regionales para clubes de lectura (16 regiones; RM: sept. 2023) | PDF con fichas: título, autor, editorial, ISBN, año, páginas, copias, género | Tabla SQL: búsqueda de títulos y disponibilidad | Interna (SNBP), pública |
| Catálogo de la Biblioteca Pública Digital para clubes: hojas *Descarga libre* (≈1200 títulos) y *Licencias* (≈2200, con cantidad de copias) | Planilla Excel | Tabla SQL: alternativas digitales | Interna (SNBP), pública |
| Clubes, integrantes, sesiones, reservas y acuerdos | Tablas | Registro del agente (escritura) y memoria | Simulados |
| Open Library / Google Books | API | Sinopsis, datos del autor, temas, obras relacionadas | Externa |
| Wikipedia en español | API | Contexto del autor y de la época | Externa |

Los archivos originales se descargan con un script (no se suben al repositorio por su tamaño).

## Restricciones

- Derechos de autor: no se indexa el texto completo de libros con derechos vigentes; solo metadatos, sinopsis y documentos públicos del SNBP.
- Datos personales de integrantes: en el proyecto se usan datos simulados.
- Las respuestas sobre préstamos y disponibilidad deben salir de los datos del catálogo, no del modelo.

## Alcance por evaluación

- **EP1 (RAG):** consultas sobre las orientaciones (RAG) y el catálogo (Text-to-SQL), enriquecidas con Open Library y Wikipedia, con citas a la fuente.
- **EP2 (agente):** herramientas para consultar (RAG, SQL, APIs externas), escribir (reservas, calendario, registro de sesiones, guía de sesión) y razonar (planificar el ciclo de lectura según páginas y frecuencia); memoria del avance de cada club; ajuste del plan cuando no hay copias suficientes o el grupo va atrasado.

## Referencias

- Sistema Nacional de Bibliotecas Públicas. (2025). *Orientaciones para mediar y gestionar clubes de lectura*. Servicio Nacional del Patrimonio Cultural. https://www.bibliotecaspublicas.gob.cl/sites/www.bibliotecaspublicas.gob.cl/files/2026-02/orientaciones-clubes-lectura%202025.pdf
- Sistema Nacional de Bibliotecas Públicas. (s.f.). *Los catálogos regionales para clubes de lectura*. https://www.bibliotecaspublicas.gob.cl/clubes-de-lectura-en-bibliotecas-publicas/los-catalogos-regionales-para-clubes-de-lectura
- Sistema Nacional de Bibliotecas Públicas. (s.f.). *Clubes de lectura de bibliotecas públicas*. https://www.bibliotecaspublicas.gob.cl/clubes-de-lectura-de-bibliotecas-publicas
