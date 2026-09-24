"""Punto de entrada de la API de comandos de voz."""

import os
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# Cargar las variables guardadas en el .env del backend. Como respaldo se
# admite el .env de la raíz del workspace, independientemente del directorio
# desde el que se lance Uvicorn.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Almacenamiento temporal en memoria para las tareas.
tasks: list[dict[str, Any]] = []
task_id_counter: int = 1


class Task(BaseModel):
    """Representa una tarea almacenada."""

    id: int
    title: str
    done: bool = False


class TaskCreate(BaseModel):
    """Datos necesarios para crear una tarea."""

    title: str
    done: Optional[bool] = False


class TaskUpdate(BaseModel):
    """Campos modificables de una tarea."""

    title: Optional[str] = None
    done: Optional[bool] = None


class InstructionRequest(BaseModel):
    """Transcripción recibida desde el frontend."""

    transcription: str


class TranscribeResponse(BaseModel):
    """Resultado completo del flujo de transcripción y ejecución."""

    transcription: str
    instruction: dict[str, Any]
    result: Any

app = FastAPI(title="Voice Command API")

SYSTEM_PROMPT = """
Eres un enrutador de comandos de voz para una API REST de gestión de tareas.
Tu única función es analizar la transcripción enviada por el usuario y
convertirla en una instrucción estructurada JSON.

ENDPOINTS DISPONIBLES Y SINTAXIS:
1. Listar tareas:
    {"endpoint": "/tasks", "method": "GET", "params": {}}
2. Crear tarea:
    {"endpoint": "/tasks", "method": "POST", "params": {"title": "<string>", "done": false}}
3. Reemplazar tarea completa por ID:
    {"endpoint": "/tasks/<id>", "method": "PUT", "params": {"title": "<string>", "done": <boolean>}}
4. Actualizar parcialmente una tarea por ID:
    {"endpoint": "/tasks/<id>", "method": "PATCH", "params": {"title": "<string>", "done": <boolean>}}
5. Eliminar tarea por ID:
    {"endpoint": "/tasks/<id>", "method": "DELETE", "params": {}}

REGLAS ESTRICTAS:
- Responde ÚNICAMENTE con el objeto JSON válido.
- No uses bloques de código Markdown.
- No agregues texto, saludo ni explicación antes o después del JSON.
"""

# Configuración de CORS para permitir peticiones desde el frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Confirma que la API está configurada correctamente."""
    return {"message": "API configurada correctamente"}


@app.get("/health")
def health() -> dict[str, str]:
    """Comprueba que el servicio está disponible."""
    return {"status": "ok"}


@app.post("/instruction", status_code=status.HTTP_200_OK)
def process_instruction(payload: InstructionRequest) -> dict[str, Any]:
    """Convierte una transcripción en una instrucción para la API de tareas."""
    if groq_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La variable GROQ_API_KEY no está configurada.",
        )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.transcription},
            ],
            # Modelo actualmente disponible en Groq para interpretación de
            # instrucciones. Puede sobreescribirse sin editar el código.
            model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_response = chat_completion.choices[0].message.content
        if not raw_response:
            raise ValueError("Groq devolvió una respuesta vacía.")
        return json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar la respuesta JSON de la IA.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en la comunicación con Groq: {exc}",
        ) from exc


def execute_instruction(instruction: dict[str, Any]) -> Any:
    """Ejecuta la instrucción JSON generada por Groq contra el CRUD local."""
    endpoint = instruction.get("endpoint")
    method = instruction.get("method", "").upper()
    params = instruction.get("params") or {}

    if endpoint == "/tasks" and method == "GET":
        return get_tasks()
    if endpoint == "/tasks" and method == "POST":
        return create_task(TaskCreate(**params))

    if isinstance(endpoint, str) and endpoint.startswith("/tasks/"):
        try:
            task_id = int(endpoint.rsplit("/", 1)[-1])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="El ID de tarea no es válido") from exc

        if method == "PUT":
            return replace_task(task_id, TaskCreate(**params))
        if method == "PATCH":
            return update_task_partial(task_id, TaskUpdate(**params))
        if method == "DELETE":
            return delete_task(task_id)

    raise HTTPException(status_code=422, detail="Instrucción de tarea no compatible")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeResponse:
    """Acepta audio o texto, interpreta el comando y ejecuta la acción."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        body = await request.json()
        transcription = InstructionRequest.model_validate(body).transcription
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file is None or not hasattr(uploaded_file, "read"):
            raise HTTPException(status_code=422, detail="Falta el archivo de audio")
        if groq_client is None:
            raise HTTPException(status_code=500, detail="La variable GROQ_API_KEY no está configurada")

        audio_bytes = await uploaded_file.read()
        language = str(form.get("language", "")).strip() or None
        transcription_result = groq_client.audio.transcriptions.create(
            file=(getattr(uploaded_file, "filename", "command.webm"), audio_bytes),
            model="whisper-large-v3-turbo",
            language=language,
            response_format="json",
        )
        transcription = str(transcription_result.text).strip()
    else:
        raise HTTPException(status_code=415, detail="Usa JSON o multipart/form-data")

    instruction = process_instruction(InstructionRequest(transcription=transcription))
    return TranscribeResponse(
        transcription=transcription,
        instruction=instruction,
        result=execute_instruction(instruction),
    )


@app.get("/tasks", status_code=status.HTTP_200_OK)
def get_tasks() -> list[dict[str, Any]]:
    """Devuelve todas las tareas almacenadas."""
    return tasks


@app.post("/tasks", status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> dict[str, Any]:
    """Crea una tarea y le asigna un ID único."""
    global task_id_counter

    new_task = {
        "id": task_id_counter,
        "title": task_data.title,
        "done": task_data.done if task_data.done is not None else False,
    }
    tasks.append(new_task)
    task_id_counter += 1
    return new_task


@app.put("/tasks/{task_id}", status_code=status.HTTP_200_OK)
def replace_task(task_id: int, task_data: TaskCreate) -> dict[str, Any]:
    """Reemplaza completamente una tarea existente."""
    for index, task in enumerate(tasks):
        if task["id"] == task_id:
            updated_task = {
                "id": task_id,
                "title": task_data.title,
                "done": task_data.done if task_data.done is not None else False,
            }
            tasks[index] = updated_task
            return updated_task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tarea con ID {task_id} no encontrada",
    )


@app.patch("/tasks/{task_id}", status_code=status.HTTP_200_OK)
def update_task_partial(task_id: int, task_data: TaskUpdate) -> dict[str, Any]:
    """Actualiza parcialmente el título o estado de una tarea."""
    for task in tasks:
        if task["id"] == task_id:
            if task_data.title is not None:
                task["title"] = task_data.title
            if task_data.done is not None:
                task["done"] = task_data.done
            return task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tarea con ID {task_id} no encontrada",
    )


@app.delete("/tasks/{task_id}", status_code=status.HTTP_200_OK)
def delete_task(task_id: int) -> dict[str, str]:
    """Elimina una tarea por su ID."""
    for index, task in enumerate(tasks):
        if task["id"] == task_id:
            tasks.pop(index)
            return {"message": f"Tarea {task_id} eliminada correctamente"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Tarea con ID {task_id} no encontrada",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
