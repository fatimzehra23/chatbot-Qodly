#!/usr/bin/env python3
# chatbot_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_llm_simple import GeminiLLM
import os
import logging
from functools import wraps

# 🔐 Configuration
GEMINI_API_KEY = "AIzaSyBLtrLx8_z4YYykZEoA90uDqIN3K5K_2TU"  
API_SECRET = "AIzaSyBLtrLx8_z4YYykZEoA90uDqIN3K5K_2TU"  

# 🚀 Initialisation de Flask
app = Flask(__name__)
CORS(app)

# 🚦 Rate Limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)
limiter.init_app(app)

# 🛡️ Authentification
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_SECRET:
            return jsonify({"error": "Authentification requise", "success": False}), 401
        return f(*args, **kwargs)
    return decorated

# 🧠 Configuration LLM et vecteurs
PROMPT_TEMPLATE = """
Tu es un assistant intelligent spécialisé dans la documentation technique de la plateforme Qodly.

Ta mission est de répondre de façon claire, concise et factuelle à la question suivante en t’appuyant uniquement sur le **contexte fourni**.

=========================
📘 CONTEXTE :
{context}
=========================

❓ QUESTION :
{question}

=========================
🎯 INSTRUCTIONS :
- Appuie-toi uniquement sur le contexte fourni
- Si aucune information pertinente n’est trouvée, dis : "Je ne trouve pas cette information dans la documentation."
- N’invente jamais une réponse
- Évite les phrases vagues ou génériques
- Sois technique si nécessaire
- Formate ta réponse avec des points ou une structure claire si le contenu est complexe
- Ne dépasse pas 300 mots

✍️ RÉPONSE :
"""

retriever = None
llm = None
is_initialized = False

# 📥 Initialiser le chatbot une seule fois
def initialize_chatbot():
    global retriever, llm, is_initialized

    if is_initialized:
        return True

    try:
        if not os.path.exists("qodly_index"):
            logger.error("Index FAISS non trouvé")
            return False

        if not GEMINI_API_KEY:
            logger.error("Clé API Gemini manquante")
            return False

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        vectorstore = FAISS.load_local(
            "qodly_index", embeddings, allow_dangerous_deserialization=True
        )

        llm = GeminiLLM(api_key=GEMINI_API_KEY)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        is_initialized = True
        logger.info("✅ Chatbot initialisé avec succès")
        return True

    except Exception as e:
        logger.error(f"Erreur d'initialisation: {e}")
        return False

# 🛠️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# 🚀 ROUTES FLASK
# =======================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if is_initialized else "not_initialized",
        "message": "Chatbot API is running"
    })

@app.route('/ask', methods=['POST'])
#@require_auth
@limiter.limit("10 per minute")
def ask_question():
    try:
        if not is_initialized and not initialize_chatbot():
            return jsonify({"error": "Chatbot non initialisé", "success": False}), 500

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Paramètre 'question' manquant", "success": False}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question vide", "success": False}), 400

        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm.invoke(prompt)

        sources = list({os.path.basename(doc.metadata.get('source', 'inconnu')) for doc in docs})

        return jsonify({
            "success": True,
            "question": question,
            "response": response,
            "sources": sources,
            "context_length": len(context)
        })

    except Exception as e:
        logger.error(f"Erreur: {e}")
        return jsonify({"error": f"Erreur interne: {e}", "success": False}), 500

@app.route('/batch', methods=['POST'])
@require_auth
def batch_questions():
    try:
        if not is_initialized and not initialize_chatbot():
            return jsonify({"error": "Chatbot non initialisé", "success": False}), 500

        data = request.get_json()
        if not data or 'questions' not in data:
            return jsonify({"error": "Paramètre 'questions' manquant", "success": False}), 400

        questions = data['questions']
        if not isinstance(questions, list):
            return jsonify({"error": "'questions' doit être une liste", "success": False}), 400

        results = []
        for i, question in enumerate(questions):
            try:
                docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = PROMPT_TEMPLATE.format(context=context, question=question)
                response = llm.invoke(prompt)
                sources = list({os.path.basename(doc.metadata.get('source', 'inconnu')) for doc in docs})
                results.append({
                    "index": i,
                    "question": question,
                    "response": response,
                    "sources": sources,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "question": question,
                    "error": str(e),
                    "success": False
                })

        return jsonify({"success": True, "results": results, "total": len(results)})

    except Exception as e:
        logger.error(f"Erreur interne: {e}")
        return jsonify({"error": f"Erreur: {e}", "success": False}), 500

@app.route('/docs', methods=['GET'])
def list_documents():
    try:
        docs_folder = "./docs"
        if not os.path.exists(docs_folder):
            return jsonify({"error": "Dossier 'docs' non trouvé", "success": False}), 404

        files = [
            {
                "name": file,
                "size": os.path.getsize(os.path.join(docs_folder, file)),
                "modified": os.path.getmtime(os.path.join(docs_folder, file))
            }
            for file in os.listdir(docs_folder) if file.endswith('.md')
        ]

        return jsonify({"success": True, "documents": files, "total": len(files)})

    except Exception as e:
        logger.error(f"Erreur: {e}")
        return jsonify({"error": f"Erreur: {e}", "success": False}), 500

# =======================
# ▶️ MAIN
# =======================
if __name__ == '__main__':
    print("🚀 Démarrage de l'API Chatbot...")
    print("📋 Endpoints disponibles:")
    print("   GET  /health  - Vérifier l'état")
    print("   POST /ask     - Poser une question")
    print("   POST /batch   - Envoyer plusieurs questions")
    print("   GET  /docs    - Lister les fichiers .md")
    print()

    if initialize_chatbot():
        print("✅ Chatbot prêt!")
    else:
        print("❌ Échec de l'initialisation")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
