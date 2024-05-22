import re
import string

preprocessed_text=[]
sentences=[]
words=[]

class tokenizer:
    def __init__(self):
        pass

    # Breaking text to sentences with pantuations like -> . ! ?
    def breakSentences(self,text):
        return re.split(r'(?<=[.!?])\s+',text)
    
    # Breaking sentences to words and punctuations
    def breakWord(self,text):
        word=re.findall(r"<\w+>|\b\w+\b|\w+(?:\'\w+)?|[^\w\s]",text)
        return word
    
    # Replacing numbers(22, 22.59, 222,000,003) to <NUM>
    def replaceNumber(self,text):
        return re.sub(r'\b\d+(,\d{3})*(\.\d+)?\b','<NUM>',text)
    
    # Replacing emails to <EMAIL>
    def replaceEmail(self,text):
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]+\b','<EMAIL>',text)
    
    # Replacing URLs to <URL>
    def replaceUrl(self,text):
        return re.sub(r'\b(?:https?://|www\.)\S+\b','<URL>',text)
    
    # Replacing hashtags to <HASHTAG>
    def replaceHashtags(self,text):
        return re.sub(r'\#\w+','<HASHTAG>',text)
    
    # Replacing mentions to <MENTION>
    def replaceMentions(self,text):
        return re.sub(r'@\S+','<MENTION>',text)
    
    # Removing punctuations
    def removePunctuation(self, text):
        return re.sub(f"(?<!<)\b[{re.escape(string.punctuation)}]\b(?!>)", '', text)

    # Adding start and end tags to sentences
    def addStartEndTags_and_RemovePunctuation(self, text,n):
        result=[]
        for t in text:
            t=[self.removePunctuation(w) for w in t if not all(char in string.punctuation for char in w)]
            sentence_with_tags = ['<SOS>'] * (n-1) + t + ['<EOS>']
            result.append(sentence_with_tags)
        return result
        

#-----------------------------------------------MAIN FUNCITON-------------------------------------------------
if __name__=="__main__":

    # path="corpus.txt"
    # with open(path) as file:
    #     input_text=file.read()
    input_text=input("Enter text:")

    token=tokenizer()
    sentences=token.breakSentences(input_text) # Sentences Tokenization


    # Preprocessing and word tokenization
    for text in sentences:
        text=token.replaceNumber(text)
        text=token.replaceEmail(text)
        text=token.replaceUrl(text)
        text=token.replaceHashtags(text)
        text=token.replaceMentions(text)
        preprocessed_text.append(text)
        text=token.breakWord(text)
        if text!=[]: # Removing empty lists (sentences with spaces only)
            words.append(text)


    #words=token.addStartEndTags_and_RemovePunctuation(words,3)
    print(words)

    