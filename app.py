import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
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
            with open('data/faq_ecommerce.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Données FAQ e-commerce enrichies
            default_faq = {
                "questions": [
                    # Salutations et politesse
                    {
                        "question": "Bonjour",
                        "reponse": "Bonjour ! 👋 Je suis ravi de vous accueillir sur notre boutique en ligne. Comment puis-je vous aider aujourd'hui ? Que ce soit pour passer une commande, suivre une livraison ou découvrir nos produits, je suis là pour vous !",
                        "tags": ["salutation", "accueil"]
                    },
                    {
                        "question": "Ça va ?",
                        "reponse": "Je vais très bien, merci de demander ! 😊 Je suis toujours prêt à vous aider avec vos achats. Et vous, comment allez-vous ? Que puis-je faire pour rendre votre expérience shopping plus agréable aujourd'hui ?",
                        "tags": ["politesse", "conversation"]
                    },
                    {
                        "question": "Merci",
                        "reponse": "De rien, c'est avec grand plaisir ! 🌟 N'hésitez surtout pas si vous avez d'autres questions. Je suis là pour vous accompagner tout au long de votre expérience d'achat. Passez une excellente journée !",
                        "tags": ["remerciement", "politesse"]
                    },
                    {
                        "question": "Bonsoir",
                        "reponse": "Bonsoir ! 🌙 Bienvenue sur notre boutique en ligne. Même en soirée, je suis disponible pour vous aider. Que recherchez-vous ce soir ?",
                        "tags": ["salutation", "accueil"]
                    },
                    
                    # Commandes
                    {
                        "question": "Comment passer une commande ?",
                        "reponse": "Passer commande est très simple ! 🛒 Voici les étapes :\n\n1. **Parcourez** notre catalogue et ajoutez vos produits préférés au panier\n2. **Cliquez** sur l'icône panier en haut à droite\n3. **Vérifiez** votre commande et cliquez sur 'Passer la commande'\n4. **Connectez-vous** ou créez un compte (c'est rapide !)\n5. **Entrez** votre adresse de livraison\n6. **Choisissez** votre mode de paiement\n7. **Validez** et c'est terminé !\n\nVous recevrez immédiatement un email de confirmation. Besoin d'aide à une étape particulière ?",
                        "tags": ["commande", "achat", "panier"]
                    },
                    {
                        "question": "Puis-je modifier ma commande ?",
                        "reponse": "Oui, mais il faut être rapide ! ⚡ Vous pouvez modifier votre commande dans les 2 heures suivant sa validation, tant qu'elle n'est pas en préparation. Connectez-vous à votre compte, allez dans 'Mes commandes' et cliquez sur 'Modifier'. Au-delà de ce délai, contactez-nous au 05 22 XX XX XX, nous ferons notre maximum pour vous aider !",
                        "tags": ["commande", "modification"]
                    },
                    {
                        "question": "Comment annuler ma commande ?",
                        "reponse": "Pour annuler une commande : ❌\n\n- **Avant expédition** : Connectez-vous, allez dans 'Mes commandes' et cliquez sur 'Annuler'. Le remboursement est automatique sous 3-5 jours.\n- **Après expédition** : Vous devrez refuser le colis à la livraison ou nous le retourner (frais de retour gratuits).\n\nPour toute urgence, appelez-nous au 05 22 XX XX XX.",
                        "tags": ["commande", "annulation"]
                    },
                    
                    # Livraison
                    {
                        "question": "Quels sont les délais de livraison ?",
                        "reponse": "Nos délais de livraison dépendent de votre ville : 📦\n\n🏙️ **Casablanca, Rabat, Marrakech** : 24-48h\n🌆 **Autres grandes villes** : 2-4 jours\n🏘️ **Zones rurales** : 3-6 jours\n\n✨ **Livraison Express disponible** : Recevez votre commande en moins de 24h pour seulement 30 DH supplémentaires (disponible dans les grandes villes).\n\nVous êtes informé par SMS à chaque étape !",
                        "tags": ["livraison", "délai", "expédition"]
                    },
                    {
                        "question": "Comment suivre ma commande ?",
                        "reponse": "Suivre votre commande est super facile ! 📍\n\n1. Connectez-vous à votre compte\n2. Cliquez sur 'Mes commandes'\n3. Sélectionnez la commande à suivre\n4. Vous verrez le statut en temps réel : préparation, expédition, en route, livré\n\nVous recevrez aussi des SMS automatiques à chaque changement de statut. Vous avez aussi un lien de suivi dans l'email de confirmation !",
                        "tags": ["livraison", "suivi", "tracking"]
                    },
                    {
                        "question": "Les frais de livraison sont de combien ?",
                        "reponse": "Nos frais de livraison sont très compétitifs ! 💰\n\n- **GRATUIT** pour toute commande de plus de 300 DH 🎉\n- **49 DH** pour les commandes inférieures à 300 DH\n- **Livraison Express** : +30 DH (livrée en moins de 24h)\n\nPetite astuce : ajoutez quelques articles pour atteindre 300 DH et profitez de la livraison gratuite !",
                        "tags": ["livraison", "frais", "prix"]
                    },
                    {
                        "question": "Livrez-vous partout au Maroc ?",
                        "reponse": "Oui, nous livrons dans tout le Maroc ! 🇲🇦\n\nDes grandes villes (Casablanca, Rabat, Tanger, Fès, Marrakech...) aux petites communes, nous couvrons l'ensemble du territoire national. Les délais varient selon votre localisation, mais nous faisons toujours au plus vite !",
                        "tags": ["livraison", "zone", "maroc"]
                    },
                    
                    # Paiement
                    {
                        "question": "Quels modes de paiement acceptez-vous ?",
                        "reponse": "Nous acceptons plusieurs modes de paiement pour votre confort : 💳\n\n✅ **Paiement à la livraison (Cash)** - Le plus populaire !\n✅ **Carte bancaire** (Visa, Mastercard, CMI)\n✅ **Virement bancaire**\n✅ **PayPal**\n✅ **Mobile Money** (Orange Money, Maroc Telecom Cash)\n\nTous nos paiements en ligne sont 100% sécurisés avec cryptage SSL. Vos données sont protégées !",
                        "tags": ["paiement", "carte", "cash"]
                    },
                    {
                        "question": "Le paiement en ligne est-il sécurisé ?",
                        "reponse": "Absolument ! 🔒 Votre sécurité est notre priorité :\n\n- Cryptage SSL 256 bits (niveau bancaire)\n- Conformité PCI-DSS\n- Aucune conservation de vos données bancaires\n- Protocole 3D Secure pour les cartes\n- Certificat de sécurité vérifié\n\nVous pouvez payer en toute confiance. Des millions de transactions sont effectuées chaque année sur notre plateforme !",
                        "tags": ["paiement", "sécurité"]
                    },
                    
                    # Retours et remboursements
                    {
                        "question": "Puis-je retourner un produit ?",
                        "reponse": "Oui, nous avons une politique de retour généreuse ! 🔄\n\n- **14 jours** pour retourner tout produit qui ne vous convient pas\n- Produit non utilisé, dans son emballage d'origine\n- **Retour GRATUIT** (nous envoyons un livreur le récupérer)\n- Remboursement sous 5-7 jours ouvrables\n\nPour initier un retour : Mon compte > Mes commandes > Demander un retour. Simple et rapide !",
                        "tags": ["retour", "remboursement", "satisfaction"]
                    },
                    {
                        "question": "Comment obtenir un remboursement ?",
                        "reponse": "Le processus de remboursement est transparent : 💵\n\n1. Demandez un retour depuis votre compte\n2. Nous récupérons le produit gratuitement\n3. Vérification du produit (sous 48h)\n4. Remboursement automatique :\n   - **Carte bancaire** : 3-5 jours ouvrables\n   - **Paiement à la livraison** : virement bancaire sous 5-7 jours\n\nVous recevez des notifications à chaque étape !",
                        "tags": ["remboursement", "retour"]
                    },
                    
                    # Produits
                    {
                        "question": "Comment trouver un produit ?",
                        "reponse": "Plusieurs façons de trouver votre bonheur ! 🔍\n\n1. **Barre de recherche** en haut : tapez le nom ou mot-clé\n2. **Catégories** dans le menu : parcourez par type de produit\n3. **Filtres** : prix, marque, couleur, taille, note...\n4. **Meilleures ventes** et **Nouveautés** sur la page d'accueil\n\nConseil : utilisez des mots simples dans la recherche pour de meilleurs résultats !",
                        "tags": ["produit", "recherche", "catalogue"]
                    },
                    {
                        "question": "Les produits sont-ils authentiques ?",
                        "reponse": "100% authentiques, garanti ! ✨\n\nNous travaillons directement avec :\n- Les marques officielles\n- Des distributeurs agréés\n- Des fournisseurs certifiés\n\nChaque produit est vérifié avant expédition. Nous ne vendons JAMAIS de contrefaçons. Votre satisfaction et votre confiance sont essentielles pour nous !",
                        "tags": ["produit", "authenticité", "qualité"]
                    },
                    {
                        "question": "Proposez-vous des promotions ?",
                        "reponse": "Oui, régulièrement ! 🎁\n\n- **Soldes saisonnières** : jusqu'à -70%\n- **Flash Sales** : chaque semaine de nouvelles offres\n- **Code promo** : pour les nouveaux clients\n- **Programme fidélité** : gagnez des points à chaque achat\n- **Newsletter** : -10% sur votre première commande\n\nInscrivez-vous à notre newsletter pour ne rien manquer !",
                        "tags": ["promotion", "réduction", "soldes"]
                    },
                    
                    # Compte et service client
                    {
                        "question": "Comment créer un compte ?",
                        "reponse": "Créer un compte est rapide et gratuit ! 👤\n\n1. Cliquez sur 'Connexion' en haut à droite\n2. Sélectionnez 'Créer un compte'\n3. Remplissez : email, mot de passe, nom\n4. Validez votre email\n5. C'est fait !\n\n**Avantages** : suivi de commandes, wishlist, adresses enregistrées, offres exclusives, points fidélité !",
                        "tags": ["compte", "inscription"]
                    },
                    {
                        "question": "Comment vous contacter ?",
                        "reponse": "Nous sommes là pour vous ! 📞\n\n- **Téléphone** : 05 22 XX XX XX (Lun-Sam, 9h-19h)\n- **WhatsApp** : 06 XX XX XX XX (réponse rapide !)\n- **Email** : contact@votremarque.ma\n- **Chat en ligne** : disponible sur le site\n- **Réseaux sociaux** : Facebook, Instagram, Twitter\n\nTemps de réponse moyen : moins de 2 heures ! Nous parlons Français, Arabe et Darija.",
                        "tags": ["contact", "service client", "assistance"]
                    },
                    {
                        "question": "Quels sont vos horaires ?",
                        "reponse": "Nous sommes disponibles presque tout le temps ! ⏰\n\n**Service client** :\n- Lundi - Samedi : 9h00 - 19h00\n- Dimanche : 10h00 - 16h00\n\n**Site web** : accessible 24h/24, 7j/7 pour commander\n**Chat en ligne** : actif pendant les horaires du service client\n\nCommandez quand vous voulez, on s'occupe du reste !",
                        "tags": ["horaires", "disponibilité"]
                    },
                    
                    # Questions produits spécifiques
                    {
                        "question": "Quelle taille choisir ?",
                        "reponse": "Pour choisir la bonne taille : 📏\n\n1. Consultez notre **guide des tailles** sur chaque fiche produit\n2. Prenez vos **mensurations** (tour de poitrine, taille, hanches)\n3. Comparez avec notre tableau\n4. En cas de doute, prenez la **taille au-dessus**\n5. Lisez les **avis clients** : ils indiquent souvent si le produit taille grand/petit\n\nRetour gratuit si la taille ne convient pas !",
                        "tags": ["taille", "vêtements", "mesures"]
                    },
                    {
                        "question": "Les couleurs sont-elles fidèles aux photos ?",
                        "reponse": "Nous faisons de notre mieux pour une représentation fidèle ! 📸\n\nNos photos sont professionnelles, mais les couleurs peuvent légèrement varier selon :\n- Votre écran\n- Les réglages de luminosité\n- L'éclairage ambiant\n\nConsultez les avis clients avec photos réelles pour vous faire une meilleure idée. Si la couleur ne vous convient pas, retour gratuit sous 14 jours !",
                        "tags": ["produit", "couleur", "photo"]
                    },
                    
                    # Garantie et SAV
                    {
                        "question": "Les produits sont-ils garantis ?",
                        "reponse": "Oui, tous nos produits sont garantis ! 🛡️\n\n- **Garantie légale** : 2 ans minimum\n- **Garantie constructeur** : selon la marque (1-3 ans)\n- **Garantie satisfait ou remboursé** : 14 jours\n\nEn cas de problème, contactez notre SAV avec votre numéro de commande. Nous gérons tout : réparation, échange ou remboursement selon le cas !",
                        "tags": ["garantie", "SAV", "après-vente"]
                    }
                ]
            }
            os.makedirs('data', exist_ok=True)
            with open('data/faq_ecommerce.json', 'w', encoding='utf-8') as f:
                json.dump(default_faq, f, ensure_ascii=False, indent=2)
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
                "reponse": "Je n'ai pas trouvé de réponse précise à votre question dans ma base de connaissances. 🤔\n\nPour vous aider au mieux, vous pouvez :\n- Reformuler votre question\n- Nous contacter directement au 05 22 XX XX XX\n- Nous écrire à contact@votremarque.ma\n- Utiliser le chat en direct sur notre site\n\nNotre équipe vous répondra avec plaisir !",
                "type": "inconnu"
            }
        
        return response

def main():
    st.set_page_config(
        page_title="ShopMart - Assistant Virtuel",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS moderne et attractif
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    .chat-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .chat-header h1 {
        color: #667eea;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .chat-header p {
        color: #666;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: messageSlideIn 0.3s ease-out;
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: messageSlideIn 0.3s ease-out;
    }
    
    .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-content {
        line-height: 1.6;
        white-space: pre-line;
    }
    
    .similarity-badge {
        display: inline-block;
        background: rgba(255,255,255,0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .tags-container {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .tag {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .input-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .stTextInput input {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .block-container {
        padding: 1rem;
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar-section h3 {
        color: #667eea;
        margin-top: 0;
        font-size: 1.2rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .quick-question {
        background: #f8f9fa;
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 3px solid #667eea;
    }
    
    .quick-question:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        color: #666;
    }
    
    .welcome-message h2 {
        color: #667eea;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Initialisation
    if 'chatbot' not in st.session_state:
        with st.spinner("🤖 Chargement de l'assistant intelligent..."):
            st.session_state.chatbot = FAQChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.markdown("""
    <div class="chat-header">
        <h1>🛍️ ShopMart Assistant</h1>
        <p>Votre assistant shopping personnel disponible 24/7</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Statistiques")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.chatbot.faq_data['questions'])}</div>
                <div class="stat-label">Questions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(st.session_state.messages)}</div>
                <div class="stat-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚡ Questions rapides")
        
        quick_questions = [
            "Comment passer une commande ?",
            "Quels sont les délais de livraison ?",
            "Comment suivre ma commande ?",
            "Puis-je retourner un produit ?",
            "Les frais de livraison ?"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                response = st.session_state.chatbot.get_response(q)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["reponse"],
                    "metadata": {
                        "score_similarite": response.get("score_similarite", 0),
                        "tags": response.get("tags", []),
                        "type": response["type"]
                    }
                })
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ℹ️ À propos")
        st.markdown("""
        Cet assistant utilise l'**intelligence artificielle** pour comprendre vos questions 
        et vous fournir des réponses précises instantanément.
        
        **Technologies :**
        - 🧠 NLP sémantique
        - 🔍 Recherche de similarité
        - ⚡ Réponses en temps réel
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Zone de chat principale
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div class="welcome-message">
                <div class="welcome-icon">🛍️</div>
                <h2>Bienvenue sur ShopMart !</h2>
                <p>Je suis votre assistant virtuel. Posez-moi toutes vos questions sur :</p>
                <p>📦 Commandes • 🚚 Livraisons • 💳 Paiements • 🔄 Retours • 🎁 Produits</p>
                <p style="margin-top: 2rem; font-size: 0.9rem; color: #999;">
                    Commencez par dire bonjour ou posez directement votre question 👇
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="message-header">👤 Vous</div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    tags_html = ""
                    if "metadata" in message and "tags" in message["metadata"] and message["metadata"]["tags"]:
                        tags = message["metadata"]["tags"]
                        tags_html = '<div class="tags-container">' + \
                                   ''.join([f'<span class="tag">#{tag}</span>' for tag in tags]) + \
                                   '</div>'
                    
                    similarity_badge = ""
                    if "metadata" in message and "score_similarite" in message["metadata"]:
                        score = message["metadata"]["score_similarite"]
                        if score > 0:
                            similarity_badge = f'<span class="similarity-badge">Confiance: {score:.0%}</span>'
                    
                    st.markdown(f"""
                    <div class="bot-message">
                        <div class="message-header">🤖 Assistant ShopMart</div>
                        <div class="message-content">{message["content"]}</div>
                        {similarity_badge}
                        {tags_html}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Zone d'input
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Votre message:",
                placeholder="Tapez votre question ici... (ex: Bonjour, Comment passer commande ?)",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("Envoyer 🚀")
        
        if submitted and user_input:
            # Ajout du message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Génération de la réponse
            with st.spinner("🔍 Recherche de la meilleure réponse..."):
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
            
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: white; margin-top: 2rem;">
        <p style="margin: 0; font-size: 0.9rem;">
            💬 Besoin d'aide humaine ? Appelez-nous au <strong>05 22 XX XX XX</strong> | 
            📧 contact@shopmart.ma
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">
            ShopMart © 2025 - Votre satisfaction est notre priorité
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()