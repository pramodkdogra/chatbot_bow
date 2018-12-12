
import os, pickle, random, nltk, string, re
#import sklearn

#import glob
#import unicodedata
#import string

from nltk import word_tokenize, pos_tag, ne_chunk


#Generating Response
#To generate a response from our bot for input questions, the concept of document similarity will be used. So we begin by importing necessary modules.
#From scikit learn library, import the TFidf vectorizer to convert a collection of raw documents to a matrix of TF-IDF features.
#Also, import cosine similarity module from scikit learn library

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity




path_="C:\\Users\\prdogra\\OneDrive - George Weston Limited-6469347-MTCAD\\prdogra\\Documents\\Sync\\MSC_Cognitive\\Year2018_2019\\COS524_Natural Language Processing\\Project_code\\BotVersion6_QandA-Cannabis\\datafiles\\"
#all_text_files = glob.glob(file_loc+'*.txt')
#print(all_text_files)

#Greetings     
ALEX_GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","hola", "whatsup", "Bonjour")
ALEX_GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "Glad to be talking to you", "I look forward to your questions on Cannabis"]
ALEX_EXIT_INPUTS = ("exit", "bye", "quit")
ALEX_THANKS=("See ya later, Bye." ,"Glad to be of Service, Bye.", "You take care now, Bye")


ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS

STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}

raw=()
sent_tokens=()
word_tokens=()
  
flag=True   
speech_flag=False
speech_ctr = 0  

def file_to_pickle():
    global raw, sent_tokens, word_tokens
    f=open(path_+"CannabisChatbotEN.txt",'r',errors = 'ignore')
    raw=f.read()
    raw=raw.lower()# converts to lowercase
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    pickle_it(sent_tokens,"sent_token")
    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    pickle_it(word_tokens,"word_token")

def pickle_it(input_,desc):
    # Dump pickled tokenizer
    out = open(path_+ desc+".pickle","wb")
    pickle.dump(input_, out)
    out.close()    

def file_read_english():
    global raw, sent_tokens, word_tokens
    if os.path.getmtime(path_+ "CannabisChatbotEN.txt") > os.path.getmtime(path_ + "sent_token.pickle"):
        f=open(path_+"CannabisChatbotEN.txt",'r',errors = 'ignore')
        raw=f.read()
        raw=raw.lower()# converts to lowercase
        sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
        pickle_it(sent_tokens,"sent_token")
        word_tokens = nltk.word_tokenize(raw)# converts to list of words
        pickle_it(word_tokens,"word_token")
    else:
        s_pickle_off = open(path_+"sent_token.pickle","rb")
        w_pickle_off = open(path_+"word_token.pickle","rb")
        
        sent_tokens = pickle.load(s_pickle_off)
        word_tokens= pickle.load(w_pickle_off)
         
#
#def file_read_french():
#    global raw, sent_tokens, word_tokens
#    f=open(path_+"Cannabis Chatbot FR",'r',errors = 'ignore')
#    raw=f.read()
#    raw=raw.lower()# converts to lowercase
#    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
#    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    



#Pre-processing the raw text. Lemmatization is the process of converting the words of a sentence to its dictionary form. 
#For example, given the words amusement, amusing, and amused, the lemma for each and all would be amuse.
#Stemming algorithms work by cutting off the end of the word, and in some cases also the beginning while looking for the root. 
#This indiscriminate cutting can be successful in some occasions, 
#but not always,


def alexGreeting(sentence):
 
    for word in sentence.split():
        if word.lower() in ALEX_GREETING_INPUTS:
            return random.choice(ALEX_GREETING_RESPONSES)

lemmert = nltk.stem.WordNetLemmatizer()

#WordNet is a semantically-oriented dictionary of English included in NLTK.
def lemmTokens(tokens):
    return [lemmert.lemmatize(token) for token in tokens]

#lambda or anonymous function 
delete_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def normalizeLemm(text):
    return lemmTokens(nltk.word_tokenize(text.lower().translate(delete_punct_dict)))

# Procedure to Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", text)
    return text         

def respondToUser(input_from_user):
    alex_bot_response=''
    TfidfVec = TfidfVectorizer(tokenizer=normalizeLemm, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        alex_bot_response=alex_bot_response+"I am sorry! I don't understand you and may not be trained enough. Can you state your question differntly ?"
        return alex_bot_response
    else:
        alex_bot_response = alex_bot_response+sent_tokens[idx]
        return alex_bot_response
        
def process():
    global word_tokens, flag, speech_flag
    #translate = YandexTranslate('trnsl.1.1.20181104T001234Z.a1af2170d545df9c.7d91bb6753fe9159bf35278d97f857e6fb86de92')
    
    while(flag==True): 
        input_from_user=input()
        input_from_user=input_from_user.lower()
        print("user input " + input_from_user)
        #if(input_from_user!='exit'):
        if (input_from_user not in ALEX_EXIT_INPUTS):
            if(input_from_user=='thanks' or input_from_user=='thank you' ):
                flag=False
                speech_flag=False
                print("ALEX: You are welcome." + random.choice(ALEX_THANKS))
            else:
                if(alexGreeting(input_from_user)!=None):
                    print("ALEX: "+alexGreeting(input_from_user))
                else:
                    sent_tokens.append(input_from_user)
                    word_tokens=word_tokens+nltk.word_tokenize(input_from_user)
                    
                    #final_words=list(set(word_tokens))
                    print("ALEX: ",end="")
                    
                    bot_response=respondToUser(input_from_user)
                    print(bot_response)
                    #sent_tokens.remove(bot_response)
                    sent_tokens.remove(input_from_user)
                    
                    
        else:
            flag=False
            speech_flag=False
            print("ALEX: Good Bye! You take care now !")
    

# =============================================================================
# def checkEnglish(text):
#     if text is None:
#         return 0
#     else:
#         text = unicode(text, errors='replace')
#         text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
#         text = text.lower()
#     words = set(nltk.wordpunct_tokenize(text))
#     if len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS):
#         return 1
#     else:
#         return 0
# =============================================================================
    

    
def main():
    global flag
    print("\n\n ALEX: Hi! My name is Alex (BOW). I will answer your queries about Cannabis Legalization in Canada. \n\n I am designed to hear your Speech. Do you want to ask your question Verbally? \n If you yes then please enter Yes and and say your question or else enter No followed by typing your question. \n\n If you want to exit this chat, type Thanks, Exit or Bye or Quit at any time!")
    
    file_read_english()
    process()
    flag=False


if __name__ == "__main__":
    main()                