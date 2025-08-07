import os
import re
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------- Nettoyage de contenu --------
def clean_text(content: str) -> str:
    content = re.sub(r'(\n\s*){3,}', '\n\n', content)
    content = re.sub(r' +\n', '\n', content)
    content = re.sub(r'[*_]{1,3}', '', content)
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
    return content.strip()

# -------- Split intelligent --------
def split_by_sections(content: str, metadata: dict) -> List[Document]:
    pattern = r'^(#{1,3})\s+(.+)$'
    lines = content.split('\n')
    sections, current_section, title, level = [], [], "", 0

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_section and len('\n'.join(current_section).strip()) > 50:
                sections.append(make_doc(title, '\n'.join(current_section), metadata, level))
            level = len(match.group(1))
            title = match.group(2).strip()
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section and len('\n'.join(current_section).strip()) > 50:
        sections.append(make_doc(title, '\n'.join(current_section), metadata, level))

    return sections or [Document(page_content=content, metadata={**metadata, 'chunk_type': 'no_sections'})]

def make_doc(title: str, content: str, metadata: dict, level: int) -> Document:
    full = f"## {title}\n\n{content.strip()}"
    return Document(
        page_content=full,
        metadata={**metadata, 'section_title': title, 'section_level': level, 'chunk_type': 'section'}
    )

def split_long_text(content: str, metadata: dict) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    chunks = splitter.split_text(content)
    return [
        Document(page_content=c.strip(), metadata={**metadata, 'chunk_index': i})
        for i, c in enumerate(chunks) if len(c.strip()) > 30
    ]

def smart_split(documents: List[Document]) -> List[Document]:
    all_chunks = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        if len(cleaned) < 100:
            continue
        sections = split_by_sections(cleaned, doc.metadata)
        if len(sections) == 1:
            sections = split_long_text(cleaned, doc.metadata)
        all_chunks.extend(sections)
    return all_chunks

# -------- Analyse --------
def summarize_chunks(chunks: List[Document]) -> None:
    print(f"📊 Total Chunks : {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"📏 Longueur moyenne : {sum(lengths) // len(lengths)} caractères")
    types = {}
    for c in chunks:
        t = c.metadata.get('chunk_type', 'unknown')
        types[t] = types.get(t, 0) + 1
    for k, v in types.items():
        print(f"🧩 {k} : {v}")
    print("\n📝 Exemple de chunk :")
    print(chunks[0].page_content[:200] + "...\n")

# -------- Indexation --------
def build_index():
    if not os.path.exists("./docs"):
        print("❌ Dossier './docs' introuvable.")
        return

    print("📥 Chargement des fichiers...")
    loader = DirectoryLoader(
        path="./docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents = loader.load()
    print(f"[✔] {len(documents)} documents chargés.")

    print("\n✂️ Découpage intelligent...")
    chunks = smart_split(documents)
    print(f"[✔] {len(chunks)} chunks générés.")
    summarize_chunks(chunks)

    print("\n🧠 Génération des embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("🗃️ Création de l’index FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("qodly_index")
    print("[✅] Index FAISS sauvegardé.")

    print("\n🔍 Test de recherche :")
    results = vectorstore.similarity_search("how to create sandbox", k=2)
    for i, r in enumerate(results):
        print(f"  Résultat {i+1} : {r.page_content[:100]}...")

    print("\n🎉 Index construit avec succès !")

# -------- Exécution --------
if __name__ == "__main__":
    build_index()
