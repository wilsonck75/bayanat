from flask import request, jsonify, current_app
from enferno.rag import rag
from enferno.rag.service import ArabicRAG
from enferno.rag.tasks import ingest_documents
from flask_security import login_required, roles_required

@rag.route('/query', methods=['POST'])
# @login_required
def query():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
        
    try:
        service = ArabicRAG()
        answer = service.query(question)
        return jsonify({"answer": answer})
    except Exception as e:
        current_app.logger.error(f"RAG Query Error: {e}")
        return jsonify({"error": str(e)}), 500

@rag.route('/ingest', methods=['POST'])
# @roles_required('admin') 
def trigger_ingest():
    """
    Trigger document ingestion task.
    Optionally pass 'data_dir' in JSON body.
    """
    data = request.get_json() or {}
    data_dir = data.get('data_dir')
    
    task = ingest_documents.delay(data_dir=data_dir)
    return jsonify({"message": "Ingestion task started", "task_id": task.id}), 202
