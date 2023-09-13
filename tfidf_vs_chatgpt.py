#!pip install spacy scikit-learn
#!python -m spacy download fr_core_news_sm

text = """
L'iPhone est une gamme de smartphone créée par Apple comprenant plusieurs générations, opérant sur le Système d'exploitation mobile iOS, développé également par Apple. Steve Jobs dévoile le premier iPhone, l'IPhone 2G, le 9 janvier 2007. Chaque année, la firme américaine publie de nouveaux modèles ainsi que des mises à jour du système d'exploitation. Au 1er novembre 2018, plus de 2,2 milliards d'iPhone ont été vendus.

Son interface utilisateur est constituée d'un écran multi-touch. Ce dernier se connecte aux Réseau de téléphonie mobile ou Wi-Fi et les utilisateurs peuvent passer des appels, naviguer sur le web, prendre des photos, écouter de la musique, envoyer et recevoir des e-mails ainsi que des SMS. Depuis son lancement, d'autres fonctionnalités sont ajoutées, notamment des écrans plus grands, la possibilité de filmer des vidéos, l'étanchéité, la possibilité d'installer des applications via l'App Store et de nombreuses fonctions d'accessibilité. Jusqu'en 2017, ils présentaient une conception avec un bouton home sur le panneau avant qui ramenait l'utilisateur à l'écran d'accueil. Depuis, les modèles sont plus coûteux et adoptent un écran frontal presque sans cadre, avec une fonction de changement d'application activée par la reconnaissance des gestes.

Le premier modèle est qualifié de révolutionnaire pour l'industrie de la téléphonie mobile et les modèles suivants suscitent également des appréciations positives. On lui attribue la popularisation du smartphone et du format ardoise, ainsi que la création d'un vaste marché pour les applications smartphone.
"""

import spacy
from spacy.tokens import Token
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

def preprocess(text):
    """Prétraitement du texte : lemmatisation, suppression des mots vides et des ponctuations"""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.pos_ != "VERB"]
    return ' '.join(tokens)

def top_k_words_with_tfidf(text, k):
    """Extraire les k mots les plus importants du texte en utilisant TF-IDF"""
    # Diviser le texte en phrases
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]

    # Prétraitement des phrases
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    
    # Appliquer TF-IDF sur les phrases prétraitées avec unigrammes et bigrammes
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2) # Ajuster le seuil de fréquence minimale (min_df)
    X = vectorizer.fit_transform(preprocessed_sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Aggréger les scores TF-IDF pour tous les documents (phrases)
    aggregated_scores = X.sum(axis=0).getA1()
    
    # Trier les scores et extraire les mots les plus importants
    sorted_indices = aggregated_scores.argsort()[::-1]
    top_words = [feature_names[i] for i in sorted_indices[:k]]
    
    return top_words



top_k_words = top_k_words_with_tfidf(text, k=10)
print(top_k_words)

# resultat >>>
# ['écran', 'iphone', 'application', 'modèle', 'utilisateur', 'fonction', 'utilisateur écran', 'mobile', 'smartphone', 'téléphonie mobile']




