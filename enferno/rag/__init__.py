from flask import Blueprint

rag = Blueprint('rag', __name__, url_prefix='/rag')

from enferno.rag import views
