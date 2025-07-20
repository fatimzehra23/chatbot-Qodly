# chatbot_qodly_fixed.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_llm_simple import GeminiLLM  # Utilise la version simple
import gradio as gr
import os

# 🔐 Configuration
GEMINI_API_KEY = "AIzaSyBLtrLx8_z4YYykZEoA90uDqIN3K5K_2TU"  

# Template pour améliorer les réponses
PROMPT_TEMPLATE = """
Tu es un assistant spécialisé dans la documentation. 
Utilise le contexte suivant pour répondre à la question de manière précise et utile.

Contexte:
{context}

Question: {question}

Instructions:
- Réponds en français
- Sois précis et concis et explique 
- Cite les sources quand c'est pertinent

Réponse:
"""

def initialize_chatbot():
    """Initialise le chatbot avec l'index FAISS et Gemini"""
    
    # Vérifier si l'index existe
    if not os.path.exists("qodly_index"):
        return None, None, "❌ Index FAISS non trouvé. Exécutez d'abord build_faiss_index.py"
    
    # Vérifier la clé API
    if not GEMINI_API_KEY or GEMINI_API_KEY == "VOTRE_CLE_API_GEMINI":
        return None, None, "❌ Clé API Gemini non configurée. Modifiez GEMINI_API_KEY dans le code."
    
    try:
        # Charger les embeddings et l'index FAISS
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.load_local(
            "qodly_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Initialiser le modèle Gemini
        llm = GeminiLLM(api_key=GEMINI_API_KEY)
        
        # Créer le retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Récupérer les 3 documents les plus pertinents
        )
        
        return retriever, llm, "✅ Chatbot initialisé avec succès"
        
    except Exception as e:
        return None, None, f"❌ Erreur lors de l'initialisation: {str(e)}"

# Initialiser le chatbot
retriever, llm, init_message = initialize_chatbot()
print(init_message)

def repondre(question, historique):
    """Fonction pour répondre aux questions"""
    
    if retriever is None or llm is None:
        return "", historique + [{"role": "user", "content": question}, {"role": "assistant", "content": "❌ Chatbot non initialisé. Vérifiez votre configuration."}]
    
    if not question.strip():
        return "", historique + [{"role": "user", "content": question}, {"role": "assistant", "content": "❓ Veuillez poser une question."}]
    
    try:
        # Rechercher des documents pertinents
        docs = retriever.get_relevant_documents(question)
        
        # Construire le contexte
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Construire le prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Générer la réponse
        reponse = llm.invoke(prompt)
        
        # Extraire les sources
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'inconnu')
            sources.append(f"📄 {os.path.basename(source)}")
        
        # Formater la réponse complète
        reponse_complete = reponse
        if sources:
            reponse_complete += f"\n\n**Sources:**\n" + "\n".join(set(sources))
        
        # Mettre à jour l'historique au format messages
        historique_updated = historique + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reponse_complete}
        ]
        
        return "", historique_updated
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de la génération de la réponse: {str(e)}"
        historique_updated = historique + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_msg}
        ]
        return "", historique_updated

def clear_chat():
    """Efface l'historique du chat"""
    return [], ""

# Interface Gradio
with gr.Blocks(title="🤖 Chatbot Documentation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Chatbot Documentation Qodly")
    gr.Markdown("Posez des questions sur votre documentation. Réponses générées avec Gemini.")
    
    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        show_label=True,
        type="messages"
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Votre question",
            placeholder="Posez votre question ici...",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("Envoyer", scale=1, variant="primary")
    
    with gr.Row():
        clear_btn = gr.Button("Effacer", scale=1)
    
    # Événements
    msg.submit(repondre, [msg, chatbot], [msg, chatbot])
    submit_btn.click(repondre, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, None, [chatbot, msg])

if __name__ == "__main__":
    # Pour Render : récupérer le port depuis la variable d'environnement
    port = int(os.environ.get("PORT", 7860))  # 7860 en fallback local

    demo.launch(
        server_name="0.0.0.0",  # Obligatoire pour Render
        server_port=port,
        show_error=True
    )