import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os
from datetime import datetime, timedelta
import random
from openai import OpenAI

class OrderTracker:
    """Système de suivi de commandes"""
    def __init__(self):
        self.orders_db = self.load_orders()
    
    def load_orders(self):
        """Charge ou crée une base de données fictive de commandes"""
        # Base de données fictive pour la démonstration
        # En production, connectez ceci à votre vraie base de données
        return {
            "CMD001": {
                "numero": "CMD001",
                "date": "2025-10-25",
                "statut": "En livraison",
                "produits": ["T-shirt Blanc", "Jean Slim"],
                "total": 450,
                "adresse": "Casablanca, Maarif",
                "transporteur": "Amana Express",
                "tracking": "AMN789456123",
                "etapes": [
                    {"date": "2025-10-25 10:30", "statut": "Commande confirmée", "termine": True},
                    {"date": "2025-10-25 14:00", "statut": "En préparation", "termine": True},
                    {"date": "2025-10-26 09:00", "statut": "Expédiée", "termine": True},
                    {"date": "2025-10-27 08:00", "statut": "En livraison", "termine": True},
                    {"date": "2025-10-28", "statut": "Livraison prévue", "termine": False}
                ]
            },
            "CMD002": {
                "numero": "CMD002",
                "date": "2025-10-26",
                "statut": "En préparation",
                "produits": ["Robe d'été", "Sandales"],
                "total": 680,
                "adresse": "Rabat, Agdal",
                "transporteur": "DHL Maroc",
                "tracking": "DHL123789456",
                "etapes": [
                    {"date": "2025-10-26 15:20", "statut": "Commande confirmée", "termine": True},
                    {"date": "2025-10-27 10:00", "statut": "En préparation", "termine": True},
                    {"date": "2025-10-28", "statut": "Expédition prévue", "termine": False}
                ]
            }
        }
    
    def get_order(self, order_number):
        """Récupère les détails d'une commande"""
        return self.orders_db.get(order_number.upper())
    
    def format_order_info(self, order):
        """Formate les informations de commande pour l'affichage"""
        if not order:
            return None
        
        # Création du suivi détaillé
        etapes_html = ""
        for etape in order["etapes"]:
            icon = "✅" if etape["termine"] else "⏳"
            style = "font-weight: bold;" if not etape["termine"] else ""
            etapes_html += f"{icon} **{etape['statut']}** - {etape['date']}\n"
        
        info = f"""
📦 **Commande #{order['numero']}**

**Statut actuel:** {order['statut']} 🚚

**Détails:**
- Date de commande: {order['date']}
- Montant total: {order['total']} DH
- Adresse de livraison: {order['adresse']}
- Transporteur: {order['transporteur']}
- N° de suivi: {order['tracking']}

**Produits commandés:**
{chr(10).join([f"  • {p}" for p in order['produits']])}

**Suivi de livraison:**
{etapes_html}

💡 Vous pouvez suivre votre colis en temps réel sur le site du transporteur avec le numéro de suivi.
        """
        return info.strip()

class FAQChatbot:
    def __init__(self, openai_api_key=None):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faq_data = self.load_faq_data()
        self.embeddings = None
        self.knn_model = None
        self.setup_embeddings()
        self.order_tracker = OrderTracker()
        
        # Configuration OpenAI
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
    def load_faq_data(self):
        """Charge les données FAQ depuis un fichier JSON"""
        default_faq = {
            "questions": [
                {
                    "question": "Bonjour",
                    "reponse": "Bonjour ! 👋 Je suis ravi de vous accueillir sur notre boutique en ligne. Comment puis-je vous aider aujourd'hui ? Que ce soit pour passer une commande, suivre une livraison ou découvrir nos produits, je suis là pour vous !",
                    "tags": ["salutation", "accueil"]
                },
                {
                    "question": "Comment passer une commande ?",
                    "reponse": "Passer commande est très simple ! 🛒 Voici les étapes :\n\n1. **Parcourez** notre catalogue et ajoutez vos produits préférés au panier\n2. **Cliquez** sur l'icône panier en haut à droite\n3. **Vérifiez** votre commande et cliquez sur 'Passer la commande'\n4. **Connectez-vous** ou créez un compte (c'est rapide !)\n5. **Entrez** votre adresse de livraison\n6. **Choisissez** votre mode de paiement\n7. **Validez** et c'est terminé !",
                    "tags": ["commande", "achat", "panier"]
                },
                {
                    "question": "Quels sont les délais de livraison ?",
                    "reponse": "Nos délais de livraison dépendent de votre ville : 📦\n\n🏙️ **Casablanca, Rabat, Marrakech** : 24-48h\n🌆 **Autres grandes villes** : 2-4 jours\n🏘️ **Zones rurales** : 3-6 jours\n\n✨ **Livraison Express disponible** : Recevez votre commande en moins de 24h pour seulement 30 DH supplémentaires.",
                    "tags": ["livraison", "délai", "expédition"]
                },
                {
                    "question": "Comment suivre ma commande ?",
                    "reponse": "Pour suivre votre commande, vous pouvez :\n\n1. Me donner votre **numéro de commande** (ex: CMD001)\n2. Vous connecter à votre compte et aller dans 'Mes commandes'\n3. Utiliser le lien de suivi envoyé par SMS/email\n\nDonnez-moi votre numéro de commande et je vous donnerai le statut en temps réel ! 📍",
                    "tags": ["livraison", "suivi", "tracking"]
                },
                {
                    "question": "Les frais de livraison sont de combien ?",
                    "reponse": "Nos frais de livraison sont très compétitifs ! 💰\n\n- **GRATUIT** pour toute commande de plus de 300 DH 🎉\n- **49 DH** pour les commandes inférieures à 300 DH\n- **Livraison Express** : +30 DH (livrée en moins de 24h)",
                    "tags": ["livraison", "frais", "prix"]
                },
                {
                    "question": "Quels modes de paiement acceptez-vous ?",
                    "reponse": "Nous acceptons plusieurs modes de paiement pour votre confort : 💳\n\n✅ **Paiement à la livraison (Cash)**\n✅ **Carte bancaire** (Visa, Mastercard, CMI)\n✅ **Virement bancaire**\n✅ **Mobile Money** (Orange Money, Cash Plus)\n\nTous nos paiements en ligne sont 100% sécurisés !",
                    "tags": ["paiement", "carte", "cash"]
                },
                {
                    "question": "Puis-je retourner un produit ?",
                    "reponse": "Oui, nous avons une politique de retour généreuse ! 🔄\n\n- **14 jours** pour retourner tout produit\n- Produit non utilisé, dans son emballage d'origine\n- **Retour GRATUIT** (nous récupérons le produit)\n- Remboursement sous 5-7 jours ouvrables",
                    "tags": ["retour", "remboursement", "satisfaction"]
                },
                {
                    "question": "Comment vous contacter ?",
                    "reponse": "Nous sommes là pour vous ! 📞\n\n- **Téléphone** : 05 22 XX XX XX (Lun-Sam, 9h-19h)\n- **WhatsApp** : 06 XX XX XX XX\n- **Email** : contact@shopmart.ma\n- **Chat en ligne** : disponible sur le site",
                    "tags": ["contact", "service client", "assistance"]
                }
            ]
        }
        return default_faq
    
    def setup_embeddings(self):
        """Crée les embeddings et le modèle KNN"""
        questions = [q["question"] for q in self.faq_data["questions"]]
        self.embeddings = self.model.encode(questions)
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(self.embeddings)
    
    def find_similar_question(self, query, threshold=0.5):
        """Trouve la question la plus similaire"""
        query_embedding = self.model.encode([query])
        distances, indices = self.knn_model.kneighbors(query_embedding)
        similarities = 1 - distances.flatten()
        
        best_match = None
        best_score = 0
        
        for similarity, idx in zip(similarities, indices[0]):
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = self.faq_data["questions"][idx]
                best_match["similarity_score"] = float(similarity)
        
        return best_match
    
    def detect_order_number(self, text):
        """Détecte un numéro de commande dans le texte"""
        import re
        # Format: CMD suivi de chiffres
        pattern = r'CMD\d+'
        matches = re.findall(pattern, text.upper())
        return matches[0] if matches else None
    
    def ask_chatgpt(self, user_question, conversation_history):
        """Utilise ChatGPT pour répondre aux questions non couvertes par la FAQ"""
        if not self.openai_client:
            return None
        
        try:
            # Contexte e-commerce pour ChatGPT
            system_prompt = """Tu es un assistant virtuel pour ShopMart, une boutique e-commerce marocaine. 
            
Informations sur l'entreprise:
- Nom: ShopMart
- Lieu: Maroc (toutes les villes)
- Produits: Vêtements, électronique, accessoires, maison
- Livraison: Partout au Maroc (gratuite > 300 DH)
- Contact: 05 22 XX XX XX, contact@shopmart.ma
- Horaires: Lun-Sam 9h-19h, Dim 10h-16h

Ta mission:
- Répondre de manière amicale et professionnelle
- Utiliser des emojis avec modération
- Être précis et concis
- Toujours proposer de l'aide supplémentaire
- Si tu ne sais pas, rediriger vers le service client

Style: Chaleureux, professionnel, orienté solution."""

            # Construction de l'historique
            messages = [{"role": "system", "content": system_prompt}]
            
            # Ajout des derniers messages pour le contexte
            for msg in conversation_history[-6:]:  # 6 derniers messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Appel à ChatGPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # ou "gpt-3.5-turbo" pour une option moins chère
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Erreur ChatGPT: {str(e)}")
            return None
    
    def get_response(self, user_input, conversation_history=None):
        """Génère une réponse basée sur l'input utilisateur"""
        # 1. Vérifier s'il y a un numéro de commande
        order_number = self.detect_order_number(user_input)
        if order_number:
            order = self.order_tracker.get_order(order_number)
            if order:
                order_info = self.order_tracker.format_order_info(order)
                return {
                    "reponse": order_info,
                    "type": "suivi_commande",
                    "tags": ["commande", "suivi"],
                    "score_similarite": 1.0
                }
            else:
                return {
                    "reponse": f"❌ Désolé, je ne trouve pas la commande **{order_number}** dans notre système.\n\nVérifiez que :\n- Le numéro est correct (format: CMDXXX)\n- La commande n'a pas été passée il y a plus de 6 mois\n\nPour plus d'aide, contactez-nous au 05 22 XX XX XX 📞",
                    "type": "commande_introuvable",
                    "tags": ["erreur", "commande"]
                }
        
        # 2. Rechercher dans la FAQ
        match = self.find_similar_question(user_input, threshold=0.5)
        
        if match and match["similarity_score"] > 0.65:
            # Bonne correspondance dans la FAQ
            return {
                "reponse": match["reponse"],
                "question_trouvee": match["question"],
                "score_similarite": match["similarity_score"],
                "tags": match.get("tags", []),
                "type": "faq"
            }
        
        # 3. Utiliser ChatGPT si disponible
        if self.openai_client and conversation_history:
            gpt_response = self.ask_chatgpt(user_input, conversation_history)
            if gpt_response:
                return {
                    "reponse": gpt_response + "\n\n💡 *Réponse générée par IA - Pour plus d'informations, contactez notre équipe.*",
                    "type": "chatgpt",
                    "tags": ["ia", "general"],
                    "score_similarite": 0.8
                }
        
        # 4. Réponse par défaut si ChatGPT n'est pas disponible
        return {
            "reponse": "Je n'ai pas trouvé de réponse précise à votre question dans ma base de connaissances. 🤔\n\nPour vous aider au mieux, vous pouvez :\n- Reformuler votre question\n- Me donner votre numéro de commande pour un suivi (ex: CMD001)\n- Nous contacter au **05 22 XX XX XX**\n- Nous écrire à **contact@shopmart.ma**\n\nNotre équipe vous répondra avec plaisir !",
            "type": "inconnu"
        }

def main():
    st.set_page_config(
        page_title="ShopMart - Assistant Virtuel",
        page_icon="🛍️",
        layout="wide"
    )
    
    # CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .chat-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 75%;
        margin-left: auto;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 75%;
    }
    
    .tag {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        margin: 0.25rem;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🛍️ ShopMart Assistant</h1>
        <p>Posez vos questions ou donnez votre numéro de commande pour le suivi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la clé API
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input(
            "Clé API OpenAI (optionnelle)",
            type="password",
            help="Entrez votre clé API OpenAI pour activer ChatGPT"
        )
        
        if api_key:
            st.success("✅ ChatGPT activé")
        else:
            st.info("ℹ️ Mode FAQ uniquement")
        
        st.markdown("---")
        st.markdown("### 📝 Exemples")
        st.markdown("""
        - "Bonjour"
        - "Comment suivre ma commande ?"
        - "CMD001" (suivi de commande)
        - "Quels sont vos horaires ?"
        """)
        
        if st.button("🗑️ Effacer conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialisation
    if 'chatbot' not in st.session_state or ('api_key' in st.session_state and st.session_state.api_key != api_key):
        st.session_state.chatbot = FAQChatbot(api_key if api_key else None)
        st.session_state.api_key = api_key
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Affichage des messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            tags_html = ""
            if "metadata" in message and "tags" in message["metadata"]:
                tags_html = ''.join([f'<span class="tag">#{tag}</span>' for tag in message["metadata"]["tags"]])
            
            st.markdown(f"""
            <div class="bot-message">
                🤖 {message["content"]}
                <div style="margin-top: 0.5rem;">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input utilisateur
    user_input = st.chat_input("Tapez votre message ou numéro de commande...")
    
    if user_input:
        # Ajouter message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Générer réponse
        response = st.session_state.chatbot.get_response(
            user_input,
            st.session_state.messages
        )
        
        # Ajouter réponse bot
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["reponse"],
            "metadata": {
                "type": response["type"],
                "tags": response.get("tags", []),
                "score_similarite": response.get("score_similarite", 0)
            }
        })
        
        st.rerun()

if __name__ == "__main__":
    main()