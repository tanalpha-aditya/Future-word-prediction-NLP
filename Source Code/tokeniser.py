import re

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

    # removing '\n'
    s = re.sub(r'\n', ' ', s)

    #Removing extra spaces
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize (s):
    
    
    s = clean(s)
    words = s.split()

    # Remove stop words
    stop_words = ["and", "the", "a", "an", "of", "in", "to"]
    words = [word for word in words if word not in stop_words]
    return words

#  to use this input a sentence in the form of list
#  and it will return tokenised and cleaned sentence back

