import re

class Tokenizer:
    def __init__(self):
        print("INIT")
    
    def sentenceTokenize(self,text):
        # handle Mrs,Dr,Mr etc
        # sentencePattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\. [A-Z])\s(\.|\?|\!)\s')
        # sentencePattern = re.compile(r"(?<!\b(?:Dr|Mr|Mrs|Ms)\.)(\. |\! |\ ?)")
        sentencePattern=re.compile(r'(?<![A-Z][a-z]\.)(?<![A-Z][a-z][a-z]\.)(?<![A-Z]\.\s[A-Z])(?<=[.!?])\s')
        sentences=re.split(sentencePattern,text)
        return sentences

    def fullTokenize(self,text):
        # sentencePattern = re.compile(r"(\. |\! |\? )")
        # sentencePattern=re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\. )(\."|\?"|\!"|\.|\?)\s*')
        wordsPattern = re.compile(r'(?<!\.\s)\s|(,|"|\.|\?|\!)')
        timePattern = re.compile(r"[0-9]+:[0-9]+")
        numberPattern = re.compile(r'\b([0-9]+(?:\.[0-9]+)?)\b')
        emailPattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+(\.[a-zA-Z]*)+")
        hashTagPattern = re.compile(r'#[a-zA-Z0-9._%+-]+')
        mentionsPattern = re.compile(r'@[a-zA-Z0-9._%+-]+')
        urlPattern=re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")
        sentences=self.sentenceTokenize(text)
        words=[]
        for sentence in sentences:
            # print("tokenizing ",sentence)

            sentence=re.sub(emailPattern,"<EMAIL>",sentence)
            sentence=re.sub(urlPattern,"<URL>",sentence)
            sentence=re.sub(mentionsPattern,"<MENTION>",sentence)
            # print(sentence)
            # print("<NUM>")
            sentence=re.sub(hashTagPattern,"<HASHTAG>",sentence)
            sentence=re.sub(timePattern,"<TIME>",sentence)
            sentence=re.sub(numberPattern,"<NUM>",sentence)

            filteredList=list(filter(lambda x:x!=None and x!="", re.split(wordsPattern,sentence)))
            # a,b=filteredList[len(filteredList)-1][:-1],filteredList[len(filteredList)-1][-1:]
            # filteredList[len(filteredList)-1]=a
            # filteredList.append(b)
            words.append(filteredList)
            # print(sentence)
        return words


if __name__=='__main__':
    # text=input("Please enter the text to be tokenized: ")
    tokenizer=Tokenizer()
    while True:
        sentence=input("Enter a sentence ")
        print(tokenizer.fullTokenize(sentence))
        

