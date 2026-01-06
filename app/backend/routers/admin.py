import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.backend.core.config import settings
from app.backend.rag.ingest import ingest_docs
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

def run_ingestion_task():
    """Background task to run ingestion."""
    logger.info("Triggering background ingestion...")
    try:
        ingest_docs()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a document and trigger re-indexing."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Ensure docs directory exists
        if not os.path.exists(settings.DOCS_DIR):
            os.makedirs(settings.DOCS_DIR)
            
        file_path = os.path.join(settings.DOCS_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File {file.filename} saved to {settings.DOCS_DIR}")
        
        # Trigger re-indexing in background
        background_tasks.add_task(run_ingestion_task)
        
        return {"message": f"File {file.filename} uploaded successfully. Ingestion started."}
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/files")
async def list_files():
    """List all uploaded documents."""
    if not os.path.exists(settings.DOCS_DIR):
        return {"files": []}
    files = [f for f in os.listdir(settings.DOCS_DIR) if f.endswith('.pdf')]
    return {"files": sorted(files)}

@router.delete("/files/{filename}")
async def delete_file(filename: str, background_tasks: BackgroundTasks):
    """Delete a specific document and trigger re-indexing."""
    file_path = os.path.join(settings.DOCS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        logger.info(f"File {filename} deleted.")
        # Trigger re-indexing to purge from vector DB
        background_tasks.add_task(run_ingestion_task)
        return {"message": f"File {filename} deleted and re-indexing started."}
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files")
async def clear_all_files(background_tasks: BackgroundTasks):
    """Delete all documents and clear the index."""
    try:
        if os.path.exists(settings.DOCS_DIR):
            shutil.rmtree(settings.DOCS_DIR)
            os.makedirs(settings.DOCS_DIR)
        
        # Also clear index immediately for better feedback
        if os.path.exists(settings.INDEX_DIR):
            shutil.rmtree(settings.INDEX_DIR)
            
        logger.info("All files and index cleared.")
        return {"message": "All files and index have been cleared."}
    except Exception as e:
        logger.error(f"Clear all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
