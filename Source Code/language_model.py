import re
import random
import nltk 
from nltk.tokenize import sent_tokenize
import math
import sys 



def clean (s):

    # lower case all words

    s = s.lower()

    # Replace urls to <URLs>
    s = re.sub(r"localhost:\d{4}\/?(?:[\w/\-?=%.]+)?|http:\/\/?localhost:\d{4}\/?(?:[\w/\-?=%.]+)?|(?:(?:https?|ftp|localhost):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+","<URL>",s)

    # Manage the Hashtags 
    s = re.sub(r"((?<![\S])#\w+(?= ))", "<HASHTAG>",s)

    # Managing the mentions
    s = re.sub(r"((?<![\S])@\w+(?= ))","<MENTION>", s)

    # Replace numbers with <NUM>
    s = re.sub(r'\b\d+(?:\.\d+)?\b', '<NUM>', s)

    # Replace currency symbols with <CURR>
    s = re.sub(r'[$€£¥]', '<CURR>', s)

    # Replace time expressions with <TIME>
    s = re.sub(r'\b\d{1,2}(?::\d{2})?\s*[APap][Mm]\b', '<TIME>', s)

    # Replace percentages with <PERCENTAGE>
    s = re.sub(r'\b\d+(?:\.\d+)?%\b', '<PERCENTAGE>', s)

    # Puntuation 
    s = re.sub(r"[^\w\s'<>]+" ,' ', s)

    #Removing extra spaces
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize (s):
    # Remove stop words ( last me hatana padega just before tokenizing)
    s = clean(s)
    words = s.split()
    # print(words)

    # stop_words = ["and", "the", "a", "an", "of", "in", "to"]
    # words = [word for word in words if word not in stop_words]
    return words

def ngramm(token, n ):
    ngram = [()]
    # counter = 0 
    for i in range(len(token)-n+1):
        ngram.append(tuple(token[i:i+n]))
    return ngram

def counter(grams):
    dict = {}
    for gram in grams:
        if gram in dict:
            dict[gram]+=1
        else:
            dict[gram]=1
    return dict

def num01(ngram, full, d):
    try:
        first_term_num = max(ngram[full]-d,0)
    except:
        first_term_num = 0
    return first_term_num

def num00(ngram, full, d):
    try:
        first_term_num = max(ngram[full]-d,0)
        first_term_num = 0.25
    except:
        first_term_num = 0
    return first_term_num

def den01(unique,ngram, history,n):
    denom = 0
    for word in unique:
        temp = ngramm(tokenize(word), 1)
        curr = temp[len(temp)-1]
        corpus = history + curr
        # print(corpus)
        # print(ngram[n])
        try:
            if(ngram[n][corpus]>0):
                denom+=ngram[n][corpus]
            else:
                continue
        except:
            denom+=0
    return denom

def den00(unique,ngram, history,n):
    denom = 0
    for word in unique:
        temp = ngramm(tokenize(word), 1)
        curr = temp[len(temp)-1]
        corpus = history + curr
        try:
            if(ngram[n][corpus]>0):
                denom+=1
            else:
                continue
        except:
            denom+=0
    return denom

def Kneser_Ney(history, ngram, check, current,unique,n):
    d = 0.10
    full = history + current
    # print(n , " ujuj ", full)
    if (n==1):
        num1 = num01(ngram[n], full, d)
        den1 = den01(unique, ngram, history, n)
        try:
            term1 = num1/den1
        except:
            term1 =0

        
        term2 = d/len(unique)
      
        return term1 + term2

    if( check ==1):
        num1 = num01(ngram[n], full, d)
        den1 = den01(unique, ngram, history, n)
        num2 = d*den00(unique, ngram, history, n)
        den2 = den01(unique,ngram, history,n)
        try:
            term1 = num1/den1
        except:
            term1 =0
        
        try:
            term2 = num2/den2
        except:
            term2 = d/len(unique)

        
    else:
        num1 = num00(ngram[n], full, d)
        den1 = den00(unique, ngram, history, n)
        num2 = d*den00(unique, ngram, history, n)
        den2 = den01(unique, ngram, history, n)
        try:
            term1 = num1/den1
        except:
            term1 =0

        try:
            term2 = num2/den2
        except:
            term2 = d/len(unique)

    history_new = history[1:]
    # print(history_new)
    return term1 + term2*Kneser_Ney(history_new, ngram,0, current,unique,n-1)

def perplexity (test,unique,ngram):

    perp = 0
    pp = 0 
    for sentence in test:
        sum = 0
        sen_token = tokenize(sentence)
        if(len(sen_token) > 4):
            for i in range(len(sen_token)-1):
                if( i > 2):
                    current = sen_token[i]
                    current_new = ngramm(tokenize(current), 1)
                    current_input = current_new[len(current_new)-1]
                    history = sen_token[i-3] + " " + sen_token[i-2] + " " + sen_token[i-1]
                    history_new = ngramm(tokenize(history), 3)
                    history_input = history_new[len(history_new)-1]

                    # prob = Kneser_Ney(history_input, ngram, 1, current_input,unique,4)
                    prob = Witten_Bell (current_input, history_input, ngram_all, 4, uniwords)

                    sum += prob*math.log2(prob)
            perp = 2**(-sum)
            # print(perp)
            pp+=perp
    return pp/len(test)

def pmlnum(ngram,n,full):
    # print(full)
    try:
        num = ngram[n][full]
    except:
        num = 0
    return num

def pmlden(ngram,n,full,history):
    den1 = 0
    try:
        den1 = ngram[n-1][history]
    except:
        den1 = 0
    den2 = 0
    try:
        den2 = ngram[n][full]
    except:
        den2 = 0
    return den1 + den2

def count(ngram,n,history):
    try:
        den1 = ngram[n-1][history]
    except:
        den1 = 1
    return den1

def N(ngram,n,current):

    count = 0 
    for history in ngram[n-1]:
        sentence = history + current
        try:
            count += ngram[n][sentence]
        except:
            count += 0
        if(count == 0):
            count = 1
    return count

def Witten_Bell (current, history, ngram, n, unique):
    full = history + current

    sum = 0 
    for word in ngram[1]:
        sum += ngram[1][word]

    if(n==1):
        try:
            d = (ngram[1][current]+1)/(sum + len(unique))
        except:
            d = 1/(sum + len(ngram[1]))
            
        return d

    pml = 0
    try:
        pml = pmlnum(ngram,n,full) / pmlden(ngram,n,full,history)
    except:
        pml = 0 
    

    lam = 0
    try:
        lam = N(ngram,n,current) / (N(ngram,n,current) + count(ngram,n,history))
        # print ( " lam ==>" ,lam)
    except:
        lam = 1/sum

    # print ( " lam ==>" ,lam)
    history_new = history[1:]
    return ( lam*pml + (1-lam)*Witten_Bell (current, history_new, ngram, n-1, unique))


if(sys.argv[1] == 'k'):
    with open(sys.argv[2], 'r') as file:
        text = file.read()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = sent_tokenize(text)

    random.shuffle(text)

    #splitting
    train = text[:len(text) - 10]
    test = text[len(text)- 10 :]

    train = " ".join(train) + " "

    with open("train_pride.txt", "w") as f:
               f.write(train)

        # Write the testing sentences to a new file
    testa = " ".join(test) + " "
    with open("test_pride.txt", "w") as f:
                f.write(testa)

    token = tokenize(train)

    four_grams = counter(ngramm(token, 4))
    three_grams = counter(ngramm(token, 3))
    two_grams = counter(ngramm(token, 2))
    one_grams = counter(ngramm(token, 1))
    ngram_all = {4: four_grams, 3: three_grams, 2: two_grams, 1:one_grams}

    uniwords = tuple(set(token))
    sen1 = input("Enter all except last word : ")
    sen2 = input("Enter last word: ")

    current = sen2
    current_new = ngramm(tokenize(current), 1)
    current_input = current_new[len(current_new)-1]
    # print (current_input)
    history = sen1
    history_new = ngramm(tokenize(history), 3)
    history_input = history_new[len(history_new)-1]
    # print (history_input)
    print(Kneser_Ney(history_input, ngram_all, 1, current_input,uniwords,4))
    # sen = input("Enter a Sentence: ")
    # print(perplexity([sen],'k',ngram_Alldict))

if(sys.argv[1] == 'w'):
    with open(sys.argv[2], 'r') as file:
        text = file.read()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = sent_tokenize(text)

    random.shuffle(text)

    #splitting
    train = text[:len(text) - 10]
    test = text[len(text)- 10 :]

    train = " ".join(train) + " "

    with open("train_pride.txt", "w") as f:
               f.write(train)

        # Write the testing sentences to a new file
    testa = " ".join(test) + " "
    with open("test_pride.txt", "w") as f:
                f.write(testa)

    token = tokenize(train)

    four_grams = counter(ngramm(token, 4))
    three_grams = counter(ngramm(token, 3))
    two_grams = counter(ngramm(token, 2))
    one_grams = counter(ngramm(token, 1))
    ngram_all = {4: four_grams, 3: three_grams, 2: two_grams, 1:one_grams}

    uniwords = tuple(set(token))
    sen1 = input("Enter all except last word : ")
    sen2 = input("Enter last word: ")

    current = sen2
    current_new = ngramm(tokenize(current), 1)
    current_input = current_new[len(current_new)-1]
    # print (current_input)
    history = sen1
    history_new = ngramm(tokenize(history), 3)
    history_input = history_new[len(history_new)-1]
    # print (history_input)
    print(Witten_Bell (current_input, history_input, ngram_all, 4, uniwords))
# with open('Pride and Prejudice - Jane Austen.txt', 'r') as file:
#     text = file.read()

# text = re.sub(r'\n', ' ', text)
# text = re.sub(r'\s+', ' ', text).strip()
# text = sent_tokenize(text)

# random.shuffle(text)

# #splitting
# train = text[:len(text) - 10]
# test = text[len(text)- 10 :]

# train = " ".join(train) + " "

# with open("train_pride.txt", "w") as f:
#         f.write(train)

# # Write the testing sentences to a new file
# testa = " ".join(test) + " "
# with open("test_pride.txt", "w") as f:
#         f.write(testa)

# token = tokenize(train)

# four_grams = counter(ngramm(token, 4))
# three_grams = counter(ngramm(token, 3))
# two_grams = counter(ngramm(token, 2))
# one_grams = counter(ngramm(token, 1))
# ngram_all = {4: four_grams, 3: three_grams, 2: two_grams, 1:one_grams}

# uniwords = tuple(set(token))

# pp = perplexity (test,uniwords,ngram_all)
# print(pp)

# current = "be"
# current_new = ngramm(tokenize(current), 1)
# current_input = current_new[len(current_new)-1]
# print (current_input)
# history = "say you will"
# history_new = ngramm(tokenize(history), 3)
# history_input = history_new[len(history_new)-1]
# print (history_input)
# print(ngram_all[4])
# ans = Witten_Bell (current_input, history_input, ngram_all, 4, uniwords)
# prob = Kneser_Ney(history_input, ngram_all, 1, current_input,uniwords,4)
# print(ans)