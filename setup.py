# setup.py
"""
Script de configuration pour le chatbot de documentation
"""

import os
import subprocess
import sys

def install_requirements():
    """Installe les dépendances"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False

def create_docs_folder():
    """Crée le dossier docs s'il n'existe pas"""
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print("📁 Dossier 'docs' créé")
        
        # Créer un fichier d'exemple
        with open("docs/exemple.md", "w", encoding="utf-8") as f:
            f.write("""# Documentation Exemple

## Introduction
Ceci est un exemple de documentation pour tester le chatbot.

## Fonctionnalités
- Recherche dans la documentation
- Réponses avec IA
- Interface utilisateur simple

## Installation
1. Installez les dépendances
2. Configurez votre clé API Gemini
3. Lancez le chatbot
""")
        print("📝 Fichier d'exemple créé dans docs/exemple.md")
    else:
        print("📁 Dossier 'docs' déjà existant")

def main():
    """Fonction principale de setup"""
    print("🚀 Configuration du chatbot de documentation")
    print("=" * 50)
    
    # Créer le dossier docs
    create_docs_folder()
    
    # Installer les dépendances
    if not install_requirements():
        print("❌ Échec de l'installation")
        return
    
    print("\n✅ Configuration terminée!")
    print("\n📋 Prochaines étapes:")
    print("1. Ajoutez vos fichiers .md dans le dossier 'docs'")
    print("2. Modifiez GEMINI_API_KEY dans chatbot_qodly.py")
    print("3. Exécutez: python build_faiss_index.py")
    print("4. Lancez: python chatbot_qodly.py")

if __name__ == "__main__":
    main()