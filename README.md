# Async TTS — Interview Prep & Event-Driven Reference Backend

Proyecto de referencia diseñado desde la perspectiva de un **Staff Engineer** para preparar una entrevista técnica en vivo (**CoderPad, 90 min, problemas de nivel medio**). Simula un pipeline real de **Text-to-Speech (TTS) asíncrono** combinando estructuras de datos clásicas, arquitectura orientada a eventos (`asyncio`) y buenas prácticas de producción en Python/FastAPI.

---

## 1. Árbol de Archivos del Proyecto

```text
DSA_python/
├── api/
│   ├── __init__.py
│   ├── routes.py                    # Endpoints POST /synthesize, GET /jobs/{id}, GET /dlq
│   └── schemas.py                   # Modelos Pydantic v2 de entrada/salida
├── core/
│   ├── __init__.py
│   ├── chunker.py                   # Algoritmo Sliding Window con ajuste por puntuación
│   ├── exceptions.py                # Jerarquía de excepciones de dominio
│   ├── parser.py                    # String Parsing + Stack para etiquetas estilo SSML
│   ├── state.py                     # Hash Maps (job_id -> JobState, request_hash -> job_id) + DLQ
│   └── voice_graph.py               # Grafo de dependencias de voces (DFS topológico + ciclos y BFS)
├── events/
│   ├── __init__.py
│   ├── bus.py                       # Event Bus en memoria sobre asyncio.Queue + pub/sub
│   └── models.py                    # Dataclasses de eventos: TTSRequested, TTSCompleted, TTSFailed
├── workers/
│   ├── __init__.py
│   └── tts_worker.py                # Worker asíncrono con reintentos, DLQ y envío de Webhooks (httpx)
├── tests/
│   ├── __init__.py
│   ├── test_api_and_idempotency.py  # Tests de integración: idempotencia, worker, webhooks y DLQ
│   ├── test_chunker.py              # Tests unitarios del Sliding Window
│   ├── test_parser.py               # Tests unitarios del parser SSML
│   └── test_voice_graph.py          # Tests unitarios de DFS/BFS y detección de ciclos
├── main.py                          # Application factory de FastAPI y ciclo de vida (lifespan)
├── pytest.ini                       # Configuración de pytest y pytest-asyncio
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Guía de ejecución, arquitectura, trade-offs y Q&A de entrevista
```

---

## 2. Cómo Ejecutar la API y los Tests

### Instalación de dependencias
```bash
python -m venv .venv
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# En Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Ejecutar los tests (`pytest`) y chequeo de tipos (`mypy`)
```bash
pytest -v
mypy .
```

### Iniciar el servidor (`uvicorn`)
```bash
uvicorn main:app --reload --port 8000
```
- Documentación interactiva Swagger UI: `http://127.0.0.1:8000/docs`

### Ejemplo rápido con `curl`
```bash
# 1. Encolar una síntesis asíncrona (devuelve 202 Accepted + job_id)
curl -X POST "http://127.0.0.1:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world! <break time=\"500ms\"/> Welcome to <emphasis level=\"strong\">ElevenLabs</emphasis> streaming TTS.",
    "voice_id": "cloned_narrator_v1"
  }'

# 2. Consultar el estado del job y sus chunks generados
curl "http://127.0.0.1:8000/jobs/<JOB_ID>"
```

---

## 3. Mapa de Algoritmos y Estructuras de Datos

| Patrón / Estructura | Archivo y Función | Complejidad | Por qué se usa en TTS |
| :--- | :--- | :--- | :--- |
| **Hash Map (`dict`)** | `core/state.py` (`JobStore`) | `O(1)` tiempo promedio | Búsqueda instantánea de estado (`job_id -> JobState`) e índice de idempotencia (`sha256(voice_id::text) -> job_id`). |
| **Sliding Window** | `core/chunker.py` (`chunk_text_sliding_window`) | `O(W)` tiempo (`W` = palabras) | Divide el texto en fragmentos de tamaño acotado para streaming de baja latencia, retrocediendo al último signo de puntuación (`.`, `!`, `?`, `;`, `:`) para no cortar frases a mitad de entonación. |
| **String Parsing + Stack** | `core/parser.py` (`parse_ssml_text`) | `O(N)` tiempo y espacio | Escanea etiquetas `<break time="..."/>` y valida con una pila explícita la apertura/cierre de `<emphasis>`, extrayendo texto limpio y offsets exactos. |
| **Graph Traversal (DFS / BFS)** | `core/voice_graph.py` (`VoiceDependencyGraph`) | `O(V + E)` tiempo y espacio | Resuelve la cadena de voces base para voces clonadas mediante **DFS Post-Order (Orden Topológico)** con coloración de 3 estados para detectar ciclos (`CyclicVoiceDependencyError`). |

---

## 4. Trade-offs de Diseño (Mentalidad Staff Engineer)

### 4.1 ¿Por qué usar `asyncio.Queue` en lugar de un broker externo (Kafka / RabbitMQ / Redis Streams)?
- **Ventajas en entrevista / prototipo / single-node**:
  - **Cero latencia de red y cero dependencias de infraestructura**: ideal para un entorno autocontenido como CoderPad o tests de integración deterministas (`await bus.join()`).
  - **Backpressure nativo**: `asyncio.Queue(maxsize=1000)` permite bloquear o rechazar productores en memoria si el worker se satura.
- **Limitaciones frente a Kafka/RabbitMQ**:
  - **Volatilidad**: si el proceso de FastAPI muere (`SIGKILL` / OOM), todos los eventos encolados en RAM se pierden.
  - **Escalado horizontal**: `asyncio.Queue` vive en un único proceso Python; no permite repartir trabajo entre múltiples contenedores/nodos GPU sin un broker distribuido.

### 4.2 ¿Cómo se maneja la idempotencia sin una base de datos externa?
- Calculamos un hash determinista `SHA-256(voice_id + "::" + text.strip())` y lo guardamos en un Hash Map `_idempotency_index: dict[str, str]` protegido por un `asyncio.Lock` para evitar *race conditions* entre corrutinas concurrentes.
- **Ciclo de vida de la clave**: mientras el job está en `pending` o `processing`, cualquier petición idéntica recibe el mismo `job_id` (`idempotent_reused=True`). Cuando el job pasa a un estado terminal (`completed` o `failed`), liberamos la entrada del índice para permitir re-síntesis futuras.
- **Trade-off**: en un clúster multi-réplica, dos peticiones idénticas balanceadas a pods distintos no compartirían el diccionario en RAM. En producción usaríamos `SET key job_id NX EX 300` en **Redis**.

### 4.3 ¿Qué limitaciones tiene el Sliding Window por palabras para textos muy largos?
1. **Materialización completa en memoria (`text.split()`)**: para un libro entero (ej. 500k palabras), crear la lista completa de strings consume memoria innecesaria de golpe. Para textos gigantes conviene un **generador lazy (`Iterator[str]`)** sobre un stream de caracteres.
2. **Longitud acústica desigual**: 8 palabras cortas (*"I am here now in the sun"*) duran mucho menos en audio que 8 palabras compuestas o números largos (*"198475938475 internationalization..."*). En sistemas TTS de producción, la ventana suele acotarse por **cantidad de fonemas/tokens** además de palabras y signos de puntuación.
3. **Oraciones sin puntuación (run-on sentences)**: si el usuario envía 200 palabras sin un solo punto o coma, el lookback de puntuación no encuentra corte natural y fuerza un corte duro en `window_size`, pudiendo degradar la prosodia si no se aplica *overlap* o *cross-fading* de audio.

---

## 5. Preguntas Típicas de Entrevista (Q&A sobre este código)

### P1: *"¿Qué pasa si el webhook del cliente falla, tarda 30 segundos o devuelve HTTP 500?"*
- **En el código actual**: el método `_send_webhook` usa un timeout estricto (`timeout=5.0`) y captura excepciones de `httpx` para que un fallo del servidor del cliente **nunca** revierta el estado `completed` del job ni tumbe el loop del worker.
- **Cómo evolucionarlo a producción**:
  1. Desacoplar el envío de webhooks a su propia cola/worker (`webhook.dispatch`) independiente del worker de GPU/síntesis.
  2. Implementar **Exponential Backoff con Jitter** (ej. reintentos a los `2s, 4s, 8s, 16s...`) ante errores `5xx` o timeouts, pero no reintentar errores `4xx` (salvo `429 Too Many Requests`).
  3. Firmar el payload con un header `X-ElevenLabs-Signature` (HMAC-SHA256) para que el cliente verifique autenticidad.

### P2: *"¿Cómo escalarías el worker si llegan 10,000 peticiones por segundo?"*
1. **Separar planos de cómputo**:
   - **Pre-procesamiento CPU-bound** (SSML parsing + chunking): ligero, se ejecuta en los nodos API o en workers ligeros.
   - **Inferencia GPU-bound** (síntesis neuronal de audio): consume de una cola particionada (Kafka / RabbitMQ / SQS) con *Consumer Groups* auto-escalados según el lag de la cola (`queue_depth / target_latency`).
2. **Concurrencia interna en Python**: en `TTSWorker`, en lugar de un único loop secuencial, lanzar un pool de `N` corrutinas consumidoras (`asyncio.TaskGroup`) o un `asyncio.Semaphore(MAX_CONCURRENT_JOBS)` para procesar múltiples jobs en paralelo mientras esperan I/O o inferencia en un servidor Triton/vLLM/TorchServe.
3. **Dynamic Batching**: agrupar chunks de distintos `job_id` que usen el mismo `base_voice_id` (resuelto por nuestro `VoiceDependencyGraph`) en un mismo batch de GPU.

### P3: *"¿Cómo evitarías duplicados en la Dead Letter Queue (DLQ) y cómo gestionarías el reprocesamiento?"*
- **Prevención de duplicados**: en `JobStore.push_to_dlq()` mantenemos un conjunto auxiliar `_dlq_job_ids: set[str]` (`O(1)` lookup) junto con la lista `_dlq` bajo el mismo `asyncio.Lock`. Si por alguna condición de carrera un evento fallido se procesara dos veces, el check `if job.job_id in self._dlq_job_ids: return` garantiza unicidad por `job_id`.
- **Reprocesamiento (Redrive)**: en producción añadiríamos un endpoint administrativo `POST /dlq/{job_id}/redrive` que reinicie `retries = 0`, cambie el estado a `pending`, elimine el registro de la DLQ y vuelva a publicar `TTSRequested` en el bus de eventos.

### P4: *"¿Qué ocurre si los diccionarios `_jobs` y `_dlq` crecen indefinidamente en memoria?"*
- Un Hash Map en memoria sin política de expulsión causará un **Memory Leak (OOM)** tras millones de jobs.
- **Solución**: implementar una política **TTL + LRU Eviction** (usando ` collections.OrderedDict` o una min-heap por `updated_at`) para purgar jobs en estado terminal (`completed` / `failed`) tras, por ejemplo, 1 hora, o persistirlos en almacenamiento frío (PostgreSQL / DynamoDB / S3).

