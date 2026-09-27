# RAG Integration Walkthrough

I have integrated the Arabic RAG pipeline into the Bayanat repository.

## Changes Made

### 1. Dependencies
Updated `pyproject.toml` to include necessary AI/ML libraries:
- `langchain`, `chromadb`, `openai`, `huggingface`
- `pypdf`, `torch`

### 2. Module Structure (`enferno/rag/`)
Created a new module for all RAG logic:
- `service.py`: Contains the `ArabicRAG` class (ported from `rag_pipeline.py`).
- `tasks.py`: Contains Celery tasks for ingestion (ported from `ingest.py`).
- `views.py`: API endpoints (`/rag/query`, `/rag/ingest`).
- `utils.py`: Helper utilities (e.g., `ImageLoader`).

### 3. Application Integration
- **Blueprint Registration**: Registered the `rag` blueprint in `enferno/app.py`.
- **Configuration**: Added RAG-related settings to `enferno/settings.py` (`CHROMA_DB_DIR`, `EMBEDDING_MODEL`).

## Verification
I created a basic integration test `tests/test_rag_integration.py` to verify the blueprint registration.

> [!NOTE]
> The automated verification step failed in the agent environment because the project dependencies (specifically `pytest` and the new AI libs) were not installed in the shell environment.

Please verify in your local environment:
1.  **Install Dependencies**:
    ```bash
    pip install ".[ai]"
    pip install pytest
    ```
2.  **Run the Test**:
    ```bash
    pytest tests/test_rag_integration.py
    ```

## Next Steps
1.  **Environment Variables**: Set `OPENAI_API_KEY` and `CHROMA_DB_DIR` in your `.env` file.
2.  **Run Ingestion**: Trigger the ingestion task via the API or celery worker.
    ```python
    # Example trigger via python shell
    from enferno.rag.tasks import ingest_documents
    ingest_documents.delay()
    ```
