import tokenizer as TK
import numpy as np
from collections import OrderedDict
import math
import sys
import random


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


#-------------------------To find lembda1, lembda2, lembda3 values by normalization-------------------
def compute_Lembdas(uniGramList,biGramList,triGramList,total_frequency):
    L1,L2,L3=0,0,0
    for tri in triGramList:
        tempL1,tempL2,tempL3=0,0,0
        # Case - 1
        seq1=tri[:2]
        if seq1 in biGramList:
            if biGramList[seq1]-1>0:
                tempL1=(triGramList[tri]-1)/(biGramList[seq1]-1)
        
        # Case - 2
        seq2=tri[1:3]
        if tri[1:2] in uniGramList:
            if uniGramList[tri[1:2]]-1>0:
                tempL2=(biGramList[seq2]-1)/(uniGramList[tri[1:2]]-1)
        
        # Case - 3
        seq3=tri[2:]
        if total_frequency-1>0:
            tempL3=(uniGramList[seq3]-1)/(total_frequency-1)
        
        max_temp=max(tempL1,tempL2,tempL3)
        if max_temp==tempL1:
            L1+=triGramList[tri]
        elif max_temp==tempL2:
            L2+=triGramList[tri]
        elif max_temp==tempL3:
            L3+=triGramList[tri]

    # Normalizing lembdas
    tot=L1+L2+L3
    L1=L1/tot
    L2=L2/tot
    L3=L3/tot
    return L1,L2,L3


#--------------------------------------------- Linear Interpolation-----------------------------------------
def LinearInterpolation(uniGramList, biGramList,triGramList,words2,L1,L2,L3,total_frequency):

    Prob=1.0
    for w in words2:
        for i in range(len(w)-3+1):
            seq=w[i:i+3]

            seq2=tuple(seq[:2])
            if seq2 not in biGramList:
                p1=0
            else:
                p1=L1*(triGramList.get(tuple(seq),0)/biGramList[seq2])
            
            seq3=tuple(seq[2:3])
            if seq3 not in uniGramList:
                p2=0
            else:     
                p2=L2*(biGramList.get(tuple(seq[1:3]),0)/uniGramList[seq3])
            
            if total_frequency==0:
                p3=0
            else:
                p3=L3*(uniGramList.get(tuple(seq[2:3]),0)/total_frequency)

            Prob=Prob*(p1+p2+p3)

    return Prob

    

#------------------------------------------------Good Turing---------------------------------------------------
def good_Turing(triGramList,words2):    

    # Calculating frequency of frequency of trigrams of corpus
    freq_of_freq={}
    for i in triGramList.values():
        if i in freq_of_freq:
            freq_of_freq[i]+=1
        else:
            freq_of_freq[i]=1


    # Calculating Zr values
    Zr={}
    #freq_of_freq=OrderedDict(sorted(freq_of_freq.items()))
    freq_of_freq_keys=list(freq_of_freq.keys())
    freq_of_freq_values=list(freq_of_freq.values())


    for i in range(len(freq_of_freq_keys)):
        if i==0:
            Zr[freq_of_freq_keys[i]]=(2*freq_of_freq_values[i])/freq_of_freq_keys[i+1]
        elif i==len(freq_of_freq)-1:
            Zr[freq_of_freq_keys[i]]=(2*freq_of_freq_values[i])/(abs(2*freq_of_freq_keys[i]-freq_of_freq_keys[i-1]))
        else:
            Zr[freq_of_freq_keys[i]]=(2*freq_of_freq_values[i])/(abs(freq_of_freq_keys[i+1]-freq_of_freq_keys[i-1]))


    # Performing Linear Regression on Zr keys and values, we will get weight and bias
    def myLinRegression(rs, zs):
        log_rs = np.log(rs)
        log_zs = np.log(zs)
        
        coef = np.polyfit(log_rs, log_zs, 1)
        weight, bias = coef
        return weight, bias


    # Smoothning values of freq_of_freq dictionary
    rs=list(Zr.keys())
    zs=list(Zr.values())
    weight,bias=myLinRegression(rs,zs)
    New_r={}

    for i in freq_of_freq:
        Nr=np.exp((weight*np.log(i))+bias)
        Nr_1=np.exp((weight*np.log(i+1))+bias)
        New_r[i]=((i+1)*Nr_1)/Nr



    # Smoothing r values
    Smooth_r_values={}
    for r in freq_of_freq:
        Nr=np.exp((weight*np.log(r))+bias)
        Nr_1=np.exp((weight*np.log(r+1))+bias)
        
        if r+1 not in freq_of_freq:
            Smooth_r_values[r]=New_r[r]
        else:
            x = (float(r+1) * freq_of_freq[r+1]) / freq_of_freq[r]

            # Finding standard deviation for comparison
            std = np.sqrt((float(r+1)**2) * (Nr_1 / Nr**2) * (1 + (Nr_1 / Nr)))

            if abs(New_r[r]-x) > std: 
                Smooth_r_values[r]=x
            else:
                Smooth_r_values[r]=New_r[r]

    

    # Finding probability of sentence
    tot=0.0
    for i,j in Smooth_r_values.items():
        tot+=j
    Prob=0.0
    p0=freq_of_freq[1]/sum(triGramList.values())
    for w in words2:
        for i in range(len(w)-3+1):
            seq=tuple(w[i:i+3])
            if seq in triGramList:
                x=triGramList[seq]
                Prob+=math.log2((Smooth_r_values[x]/tot))
            else:
                Prob+=math.log2(p0)

    Prob=math.pow(2,Prob)
    return Prob                


#-------------------------------------------Finding Perplexity----------------------------------
# Perplexity of Linear Interpolation
def perplexity_LI(uniGramList,biGramList,triGramList,ww,L1,L2,L3,total_frequency):
    Output_File1={}
    Perp=0
    for w in ww:
        Prob=LinearInterpolation(uniGramList,biGramList,triGramList,w,L1,L2,L3,total_frequency)
        if Prob==0:
            Prob=math.e**(-9)
        Output_File1[tuple(w)]=Prob
        Perp+=(1/Prob)**(1/len(w))
    Perp=Perp/len(ww)
    return Perp,Output_File1


# Perplexity of Good Turing
def perplexity_GT(triGramList, ww):
    Output_File2={}
    Perp=0
    for w in ww:
        Prob=good_Turing(triGramList,w)
        Output_File2[tuple(w)]=Prob
        Perp+=(1/Prob)**(1/len(w))
    return Perp,Output_File2





#-------------------------------------------Main Functions--------------------------------------
if __name__=='__main__':

    model_type=sys.argv[1]
    path=sys.argv[2]
    inpt=input("Input Sentence: ")

    preprocessed_text=[]
    preprocessed_text2=[]
    preprocessed_text3=[]
    sentences=[]
    sentences2=[]
    sentences3=[]
    words=[]
    words2=[]
    words3=[]
    uniGram={}
    biGram={}
    triGram={}


    with open(path) as file:
        input_text=file.readlines()
    
    random.shuffle(input_text)

    X_train = input_text[:(int(0.9*len(input_text)))]
    X_test = input_text[(int(0.9*len(input_text))):]

    input_text = ''.join(X_train)
    input_text2 = ''.join(X_test)

    #Creating instance of Tokenizer class
    token=TK.tokenizer()

    # Tokenizing the training text
    sentences=token.breakSentences(input_text)
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


    # Tokenizing the test text
    sentences3=token.breakSentences(input_text2)
    for text in sentences3:
            text=token.replaceNumber(text)
            text=token.replaceEmail(text)
            text=token.replaceUrl(text)
            text=token.replaceHashtags(text)
            text=token.replaceMentions(text)
            preprocessed_text3.append(text)
            text=token.breakWord(text)
            if text!=[]: # Removing empty lists (sentences with spaces only)
                words3.append(text)


    # Tokenizing the input sentence
    sentences2=token.breakSentences(inpt)
    for text in sentences2:
            text=token.replaceNumber(text)
            text=token.replaceEmail(text)
            text=token.replaceUrl(text)
            text=token.replaceHashtags(text)
            text=token.replaceMentions(text)
            preprocessed_text2.append(text)
            text=token.breakWord(text)
            if text!=[]: # Removing empty lists (sentences with spaces only)
                words2.append(text)

    
    # Removing punctuation and adding SOS and EOS tags
    words_1=token.addStartEndTags_and_RemovePunctuation(words,1)
    words_2=token.addStartEndTags_and_RemovePunctuation(words,2)
    words_3=token.addStartEndTags_and_RemovePunctuation(words,3) 

    words=token.addStartEndTags_and_RemovePunctuation(words,3) #Removing punctuation and adding SOS and EOS tags
    
    # Creating instances of NGram class
    uniGramobj = nGram(1)
    uniGramList = dict(uniGramobj.createNGram(words_1))

    biGramobj = nGram(2)
    biGramList = dict(biGramobj.createNGram(words_2))

    triGramobj = nGram(3)
    triGramList = dict(triGramobj.createNGram(words_3))

    # Total frequency will be the sum of all values of unigram dictionary
    total_frequency=sum(uniGramList.values())

    # Finding lembda values, L1 is trigram Lembda, L2 is Bigram Lembda, L3 is unigramram Lembda
    L1,L2,L3=compute_Lembdas(uniGramList, biGramList, triGramList, total_frequency)
    #print(L1,L2,L3)

    if model_type=="i":
        Per,Output_File1=perplexity_LI(uniGramList,biGramList,triGram,words,L1,L2,L3,total_frequency)
        #print("Perplexity of Train:",Per)

        with open("LM1_Corpus1_Train_Perplexity.txt", "a") as fileout:
            fileout.write(f"Overall Perplexity : {Per}\n")
            for i,j in Output_File1.items():
                fileout.write(f"{i}: {j}\n")
        
        Per,Output_File1=perplexity_LI(uniGramList,biGramList,triGram,words3,L1,L2,L3,total_frequency)
        #print("Perplexity of Test:",Per)
        
        with open("LM1_Corpus1_Test_Perplexity.txt", "a") as fileout:
            fileout.write(f"Overall Perplexity : {Per}\n")
            for i,j in Output_File1.items():
                fileout.write(f"{i}: {j}\n")

        P=LinearInterpolation(uniGramList,biGramList,triGram,words2,L1,L1,L3,total_frequency)
        print(P)
    else:
        Per,Output_File2=perplexity_GT(triGramList,words)
        #print("Perplexity of Train",Per)

        with open("LM2_Corpus1_Train_Perplexity.txt", "a") as fileout:
            fileout.write(f"Overall Perplexity : {Per}\n")
            for i,j in Output_File2.items():
                fileout.write(f"{i}: {j}\n")
        
        Per,Output_File2=perplexity_GT(triGramList,words3)
        #print("Perplexity of Test",Per)

        with open("LM2_Corpus1_Test_Perplexity.txt", "a") as fileout:
            fileout.write(f"Overall Perplexity : {Per}\n")
            for i,j in Output_File2.items():
                fileout.write(f"{i}: {j}\n")
        P=good_Turing(triGram,words2)
        print(P)
            