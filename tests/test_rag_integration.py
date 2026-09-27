import pytest
from enferno.app import create_app
from enferno.rag import rag

def test_rag_blueprint_registration():
    app = create_app()
    assert 'rag' in app.blueprints
    
    # Check url map
    rules = [str(rule) for rule in app.url_map.iter_rules()]
    assert any('/rag/query' in rule for rule in rules)
    assert any('/rag/ingest' in rule for rule in rules)

def test_rag_service_import():
    from enferno.rag.service import ArabicRAG
    assert ArabicRAG is not None
