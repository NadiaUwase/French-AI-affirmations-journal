import string

import nltk
import pandas as pd
import requests
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import joblib
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
global english_translated_text
global journal
global anger

# title of the application
st.write("""
# JAFFIRM
## Journale intime avec des affirmations personalise

### Des affirmations pour vous aider dans des moments difficile et vous rapellez que tout ira bien, pour vous aidez a cultiver une mentalite plus positive
""")
journal_entry = st.text_input('Comment allez vous?, comment te sens tu?')

# translating the journal entries from french to english and then
# creating a dataframe for it, cleaning it and creating a dataframe for it
if st.button('submit'):
    def translate_text(text, target_language):
        api_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=fr&tl=" + target_language + "&dt=t&q=" + text
        response = requests.get(api_url)
        if response.status_code == 200:
            translated_text = response.json()[0][0][0]
            return translated_text
        else:
            return "Error: translation failed"



    english_translated_text = translate_text(journal_entry, 'en')

        # creating a dataframe for the journal entries
        # creating a dataframe for the journal entries and appending to it the journal entries
    journal = pd.DataFrame(columns=['Raw journal entries', 'Cleaned journal entries'])
    raw_journal_entries = []
    raw_journal_entries.append(english_translated_text)

    for x in raw_journal_entries:
        # cleaning the raw input
        cleaned_text = x.strip().lower().replace('\n', ' ')
        journal = journal.append({'Raw journal entries': raw_journal_entries, 'Cleaned journal entries': cleaned_text},
                                 ignore_index=True)

        #turn the text into lower case
        #it was done above in the introductory text cleaning
        #remove punctuation and special characters
    #loading the emotions dataset for creating the count vectorizer/ bag of words model
    emotions_dataset = pd.read_csv("data/tweets_emotions_dataset.csv")
    emotions_dataset['cleaned text']= emotions_dataset['cleaned text'].values.astype('U')

    def remove_punctuations(text):
        punctuations = string.punctuation
        return text.translate(str.maketrans('', '', punctuations))
    journal['Cleaned journal entries'] = journal['Cleaned journal entries'].apply(lambda x: remove_punctuations(x))



    #remove stopwords
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        return " ".join([word for word in text.split() if word not in STOPWORDS])
    journal['Cleaned journal entries'] = journal['Cleaned journal entries'].apply(lambda x: remove_stopwords(x))

    #lemmatize the text
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

    def lemmatize_words(text):
            # find pos tags
        pos_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
        wordnet.NOUN

    journal['Cleaned journal entries'] = journal['Cleaned journal entries'].apply(lambda x: lemmatize_words(x))

    #remove all the special characters
    def remove_spl_chars(text):
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text
    journal['Cleaned journal entries'] = journal['Cleaned journal entries'].apply(lambda x: remove_spl_chars(x))
    #preprocess the text
    #apply the count vectorizer/ bag of words model
    #applying count vectorizer/bag of words on the journal entry for prediction and then categorising it

    x_features = emotions_dataset['cleaned text']
    y_labels = emotions_dataset['Emotions_encoded']
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, random_state=42)

    #import the model using joblib
    emotions_det_model = joblib.load(open('model/regression_model.joblib','rb'))
    #count vectoriser
    #def predict(journal):
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(emotions_dataset['cleaned text'])
    count_values = count_vect.transform(journal['Cleaned journal entries'].values.astype('U'))  ## Even astype(str) would work

#make an emotion dete#tion by applying the model to the text
    journal_pred = count_vect.transform(journal['Cleaned journal entries'].values.astype('U'))
    journal['emotion detected'] = emotions_det_model.predict(journal_pred)

#loading all the affirmations datasets for the different emotions
    anger = pd.read_csv('data/anger.csv')
    disgust = pd.read_csv('data/disgust.csv')
    fear = pd.read_csv('data/fear.csv')
    joy = pd.read_csv('data/joy.csv')
    neutral = pd.read_csv('data/neutral.csv')
    sadness = pd.read_csv('data/sadness.csv')
    shame = pd.read_csv('data/shame.csv')
    suprise = pd.read_csv('data/suprise.csv')

# recommend an affirmation depending on the emotion detected in the text

    if journal['emotion detected'].item() == 2:
        st.write(fear.sample(axis=0))
    elif journal['emotion detected'].item() == 0:
        st.write(anger.sample(axis=0))
    elif journal['emotion detected'].item() == 1:
        st.write(disgust.sample(axis=0))
    elif journal['emotion detected'].item() == 3:
        st.write(joy.sample(axis=0))
    elif journal['emotion detected'].item() == 4:
        st.write(neutral.sample(axis=0))
    elif journal['emotion detected'].item() == 5:
        st.write(sadness.sample(axis=0))
    elif journal['emotion detected'].item() == 6:
        st.write(shame.sample(axis=0))
    elif journal['emotion detected'].item() == 7:
        st.write(suprise.sample(axis=0))
    else:
        st.write(neutral.sample(axis=0))





# show it

#create a user account/profile

#improve the UI design


