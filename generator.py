import tokenizer as TK
import language_model as SI
import sys



#-------------------------------------------nGram class-------------------------------------------
class nGram:
    def __init__(self,n=1):
        self.n=n
    
    def createNGram(self,words):
        nGrams = {}
        for text in words:
            for i in range(len(text)-self.n+1):
                seq=tuple(text[i:i+self.n])
                if seq in nGrams:
                    nGrams[seq] += 1
                else:
                    nGrams[seq] = 1 
        return nGrams



#--------------------------------------Generation using NGram model---------------------------------------------
def gen_using_Ngram():
    list_of_gen_words_4Gram={}
    list_of_gen_words_3Gram={}
    list_of_gen_words_2Gram={}
    
    

    # -----------------Generation using 2Gram model---------------
    if tuple(last_word) not in uniGramList:
        print()
        print("Unable to generate next word using 2-Gram Model.")

    else:
        for i in uniGramList:
            new_bigram=tuple(last_word)+tuple(i)

            if new_bigram in biGramList:
                if i not in list_of_gen_words_2Gram:
                    list_of_gen_words_2Gram[i]=biGramList[new_bigram]/uniGramList[tuple(last_word)]
                else:
                    list_of_gen_words_2Gram[i]+=biGramList[new_bigram]/uniGramList[tuple(last_word)]
            
        list_of_gen_words_2Gram = dict(sorted(list_of_gen_words_2Gram.items(), key=lambda item: item[1], reverse=True))
        
        # Printing top k words
        print()
        print("Top K words with 2-Gram model:")
        for idx, (key, val) in enumerate(list_of_gen_words_2Gram.items()):
            if idx < k:
                print(key[0],":",val)
        
        if len(list_of_gen_words_2Gram)<k:
            print("Unable to generate other words.")



    #---------------Generation using 3Gram model---------------
    if tuple(last_two_words) not in biGramList:
        print()
        print("Unable to generate next word using 3-Gram Model.")

    else:
        for i in uniGramList:
            new_trigram=tuple(last_two_words)+tuple(i)

            if new_trigram in triGramList:
                if i not in list_of_gen_words_3Gram:
                    list_of_gen_words_3Gram[i]=triGramList[new_trigram]/biGramList[tuple(last_two_words)]
                else:
                    list_of_gen_words_3Gram[i]+=triGramList[new_trigram]/biGramList[tuple(last_two_words)]
            
        list_of_gen_words_3Gram = dict(sorted(list_of_gen_words_3Gram.items(), key=lambda item: item[1], reverse=True))
        
        # Printing top k words
        print()
        print("Top K words using 3-Gram model:")
        for idx, (key, val) in enumerate(list_of_gen_words_3Gram.items()):
            if idx < k:
                print(key[0],":",val)
        
        if len(list_of_gen_words_3Gram)<k:
            print("Unable to generate other words.")



    #---------------Generation using 4Gram model---------------
    if tuple(last_three_words) not in triGramList:
        print()
        print("Unable to generate next word using 4-Gram Model.")

    else:
        for i in uniGramList:
            new_quadgram=tuple(last_three_words)+tuple(i)

            if new_quadgram in quadGramList:
                if i not in list_of_gen_words_4Gram:
                    list_of_gen_words_4Gram[i]=quadGramList[new_quadgram]/triGramList[tuple(last_three_words)]
                else:
                    list_of_gen_words_4Gram[i]+=quadGramList[new_quadgram]/triGramList[tuple(last_three_words)]
            
        list_of_gen_words_4Gram = dict(sorted(list_of_gen_words_4Gram.items(), key=lambda item: item[1], reverse=True))
        
        # Printing top k words
        print()
        print("Top K words with 4-Gram model:")
        for idx, (key, val) in enumerate(list_of_gen_words_4Gram.items()):
            if idx < k:
                print(key[0],":",val)
        
        if len(list_of_gen_words_4Gram)<k:
            print("Unable to generate other words.")




#-----------------------------------Generation using LinearInterpolation----------------------------------
def gen_using_Interpolation():
    list_of_gen_words_LI={}
    
    for i in uniGramList:
        words2=[]
        words2.append(list((tuple(last_two_words)+tuple(i))))
        kk=SI.LinearInterpolation(uniGramList,biGramList,triGramList,words2,L1,L2,L3,total_frequency)

        if i not in list_of_gen_words_LI:
            list_of_gen_words_LI[i]=kk
        else:
            list_of_gen_words_LI[i]+=kk

    list_of_gen_words_LI = dict(sorted(list_of_gen_words_LI.items(), key=lambda item: item[1], reverse=True))

    # Printing top k words
    print()
    print("Top K words")
    for idx, (key, val) in enumerate(list_of_gen_words_LI.items()):
        if idx < k:
            print(key[0],":",val)
        
    if len(list_of_gen_words_LI)<k:
        print("Unable to generate other words.")




# ------------------------------------------------Main Function------------------------------------------------
if __name__=='__main__':
    
    model_type=sys.argv[1]
    path=sys.argv[2]
    k=int(sys.argv[3])
    inpt=input("Input Sentence: ")
    inpt=inpt.split(' ')
    last_two_words=[]
    preprocessed_text=[]
    sentences=[]
    words=[]
    uniGram={}
    biGram={}
    triGram={}

    last_three_words=inpt[-3:]
    last_two_words=inpt[-2:]
    last_word=inpt[-1:]

    with open(path) as file:
        input_text=file.read()

    #Creating instance of Tokenizer class
    token=TK.tokenizer()

    #----------------------------------------Tokenizing the text---------------------------------------------
    sentences=token.breakSentences(input_text)
    for text in sentences:
            text=token.replaceNumber(text)
            text=token.replaceEmail(text)
            text=token.replaceUrl(text)
            text=token.replaceHashtags(text)
            text=token.replaceMentions(text)
            preprocessed_text.append(text)
            text=token.breakWord(text)
            if text!=[]: #------------------Removing empty lists (sentences with spaces only)-----------------
                words.append(text)


    # Creating instances of NGram class
    uniGram = nGram(1)
    uniGramList = dict(uniGram.createNGram(words))

    biGram = nGram(2)
    biGramList = dict(biGram.createNGram(words))

    triGram = nGram(3)
    triGramList = dict(triGram.createNGram(words))

    quadGram = nGram(4)
    quadGramList = dict(quadGram.createNGram(words))

    if model_type == "i":
         # Total frequency will be the sum of all values of unigram dictionary
        total_frequency=sum(uniGramList.values())

        # Finding lembda values, L1 is trigram Lembda, L2 is Bigram Lembda, L3 is unigramram Lembda
        L1,L2,L3=SI.compute_Lembdas(uniGramList, biGramList, triGramList, total_frequency)
        gen_using_Interpolation()
    else:
        gen_using_Ngram()