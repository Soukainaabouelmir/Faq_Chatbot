import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os
from datetime import datetime

class FAQChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faq_data = self.load_faq_data()
        self.embeddings = None
        self.knn_model = None
        self.setup_embeddings()
        
    def load_faq_data(self):
        """Charge les données FAQ depuis un fichier JSON"""
        try:
            with open('data/faq.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Données FAQ par défaut
            default_faq = {
                "questions": [
                    {
                        "question": "Quels sont vos horaires d'ouverture ?",
                        "reponse": "Nous sommes ouverts du lundi au vendredi de 9h à 18h.",
                        "tags": ["horaires", "contact", "heures"]
                    },
                    {
                        "question": "Comment puis-je vous contacter ?",
                        "reponse": "Vous pouvez nous contacter par email à contact@entreprise.com ou par téléphone au 01 23 45 67 89.",
                        "tags": ["contact", "email", "téléphone"]
                    },
                    {
                        "question": "Quels services proposez-vous ?",
                        "reponse": "Nous proposons du développement web, de l'intelligence artificielle et du consulting digital.",
                        "tags": ["services", "offre", "produits"]
                    },
                    {
                        "question": "Acceptez-vous les cartes de crédit ?",
                        "reponse": "Oui, nous acceptons Visa, MasterCard et American Express.",
                        "tags": ["paiement", "carte", "crédit"]
                    },
                    {
                        "question": "Proposez-vous des formations ?",
                        "reponse": "Oui, nous proposons des formations en ligne et en présentiel.",
                        "tags": ["formations", "cours", "apprentissage"]
                    }
                ]
            }
            # Créer le dossier data s'il n'existe pas
            os.makedirs('data', exist_ok=True)
            with open('data/faq.json', 'w', encoding='utf-8') as f:
                json.dump(default_faq, f, ensure_ascii=False, indent=2)
            return default_faq
    
    def setup_embeddings(self):
        """Crée les embeddings et le modèle KNN"""
        questions = [q["question"] for q in self.faq_data["questions"]]
        self.embeddings = self.model.encode(questions)
        
        # Création du modèle KNN
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(self.embeddings)
    
    def find_similar_question(self, query, threshold=0.6):
        """Trouve la question la plus similaire"""
        query_embedding = self.model.encode([query])
        
        # Recherche des plus proches voisins
        distances, indices = self.knn_model.kneighbors(query_embedding)
        
        # Conversion distance -> similarité
        similarities = 1 - distances.flatten()
        
        best_match = None
        best_score = 0
        
        for similarity, idx in zip(similarities, indices[0]):
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = self.faq_data["questions"][idx]
                best_match["similarity_score"] = float(similarity)
        
        return best_match
    
    def get_response(self, user_input):
        """Génère une réponse basée sur l'input utilisateur"""
        match = self.find_similar_question(user_input)
        
        if match:
            response = {
                "reponse": match["reponse"],
                "question_trouvee": match["question"],
                "score_similarite": match["similarity_score"],
                "tags": match.get("tags", []),
                "type": "faq"
            }
        else:
            response = {
                "reponse": "Je n'ai pas trouvé de réponse précise à votre question. Pouvez-vous la reformuler ou contactez-nous directement pour plus d'informations ?",
                "type": "inconnu"
            }
        
        return response

class ChatMemory:
    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.conversation = []
    
    def add_message(self, role, content, metadata=None):
        """Ajoute un message à la mémoire"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation.append(message)
        
        # Garder seulement les derniers messages
        if len(self.conversation) > self.max_messages:
            self.conversation = self.conversation[-self.max_messages:]
    
    def get_conversation_context(self):
        """Retourne le contexte de conversation"""
        return self.conversation[-5:]  # Derniers 5 messages

def main():
    st.set_page_config(
        page_title="Chatbot FAQ Intelligent",
        page_icon="",
        layout="wide"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: right;
        max-width: 70%;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
        text-align: left;
        max-width: 70%;
        border: 1px solid #e0e0e0;
    }
    .similarity-score {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
        margin-top: 5px;
    }
    .tags {
        font-size: 0.8em;
        color: #4285f4;
        margin-top: 5px;
    }
    .stButton button {
        width: 100%;
        background-color: #4285f4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title(" Chatbot FAQ Intelligent")
    st.markdown("**Posez-moi vos questions !** Je comprends même si vous les formulez différemment grâce à l'IA sémantique.")
    
    # Initialisation
    if 'chatbot' not in st.session_state:
        with st.spinner("Chargement du modèle IA..."):
            st.session_state.chatbot = FAQChatbot()
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ChatMemory()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹÀ propos")
        st.markdown("""
        Ce chatbot utilise :
        - **Embeddings sémantiques** pour comprendre le sens
        - **Recherche de similarité** pour trouver les réponses
        - **Mémoire de conversation** pour le contexte
        """)
        
        st.header(" Statistiques")
        st.write(f"**Questions dans la base :** {len(st.session_state.chatbot.faq_data['questions'])}")
        st.write(f"**Messages échangés :** {len(st.session_state.messages)}")
        
        if st.button(" Effacer l'historique"):
            st.session_state.messages = []
            st.session_state.memory = ChatMemory()
            st.rerun()
        
        # Aperçu de la base FAQ
        st.header("📋 Questions disponibles")
        for i, q in enumerate(st.session_state.chatbot.faq_data["questions"][:5], 1):
            st.write(f"{i}. {q['question']}")
    
    # Zone de chat
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        chat_container = st.container()
        
        # Affichage des messages
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="user-message">'
                        f'<div>👤 <strong>Vous</strong></div>'
                        f'<div>{message["content"]}</div>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    bot_content = f'<div>🤖 <strong>Assistant</strong></div><div>{message["content"]}</div>'
                    
                    if "metadata" in message and "score_similarite" in message["metadata"]:
                        score = message["metadata"]["score_similarite"]
                        bot_content += f'<div class="similarity-score">Confiance: {score:.2%}</div>'
                    
                    if "metadata" in message and "tags" in message["metadata"] and message["metadata"]["tags"]:
                        tags = " | ".join(message["metadata"]["tags"])
                        bot_content += f'<div class="tags">Tags: {tags}</div>'
                    
                    st.markdown(
                        f'<div class="bot-message">{bot_content}</div>', 
                        unsafe_allow_html=True
                    )
    
    # Input utilisateur
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Votre question:",
                placeholder="Ex: Quels sont vos horaires ? Comment vous contacter ?...",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("🚀 Envoyer")
        
        if submitted and user_input:
            # Ajout du message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.memory.add_message("user", user_input)
            
            # Génération de la réponse
            with st.spinner("🔍 Recherche dans la base de connaissances..."):
                response = st.session_state.chatbot.get_response(user_input)
            
            # Ajout de la réponse du bot
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["reponse"],
                "metadata": {
                    "score_similarite": response.get("score_similarite", 0),
                    "question_trouvee": response.get("question_trouvee", ""),
                    "tags": response.get("tags", []),
                    "type": response["type"]
                }
            })
            
            st.session_state.memory.add_message("assistant", response["reponse"], {
                "score_similarite": response.get("score_similarite", 0),
                "type": response["type"]
            })
            
            st.rerun()

if __name__ == "__main__":
    main()