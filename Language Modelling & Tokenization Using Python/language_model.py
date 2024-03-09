from tokenizer import Tokenizer
import sys
from numpy import exp, log, sqrt,array,argmax
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
class lm:
    def __init__(self,corpus_path,n=3):
        self.vocab=set()
        self.n=n
        self.corpus_path=corpus_path
        self.tokenizer = Tokenizer()
        self.probabilities={}
        self.gt_probabilities={}
        self.i_weights=None
        self.i_probabilities={}
        self.train_set=None
        self.test_set=None
        self.punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
                             '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    def train(self,corpus_path):
        n_gram_model=self.generate_n_gram_model(self.n,self.n-1,corpus_path)
        self.probabilities=self.generate_initial_probabilities(n_gram_model)
        self.good_turing(n_gram_model)

        #linear interpolation - genterate n_gram_model with counts from 1 to n grams
        n_n_gram_model={}
        for i in range(1,3):
            n_n_gram_model[i]=self.generate_n_gram_model(i,2,corpus_path)
        n_n_gram_model[3]=n_gram_model

        self.linear_interpolation(n_n_gram_model)

    def linear_interpolation(self,n_n_gram_model):
        weights=[0,0,0]
        trigram_model=n_n_gram_model[3]
        bigram_model=n_n_gram_model[2]
        unigram_model=n_n_gram_model[1]
        unigrams_cnt=sum(unigram_model[("")].values())
        for context in trigram_model:
            w1=context[0]
            w2=context[1]
            for w3 in trigram_model[context]:
                f123=trigram_model[context][w3]
                f12=bigram_model[(w1,)][w2]
                f23=bigram_model[(w2,)][w3]
                f2=unigram_model[("")][w2]
                f3=unigram_model[("")][w3]
                func_vals=[(f123-1)/(f12-1) if (f12-1)>0 else 0,(f23-1)/(f2-1) if (f2-1)>0 else 0,(f3-1)/(unigrams_cnt-1)]
                maxi=argmax(func_vals)
                weights[maxi]+=1
        wts_sum=sum(weights)
        weights =[ x/wts_sum for x in weights]
        self.i_weights=weights
        self.i_probabilities={}
        self.i_probabilities[3]=self.probabilities
        self.i_probabilities[2]=self.generate_initial_probabilities(n_n_gram_model[2])
        self.i_probabilities[1]=self.generate_initial_probabilities(n_n_gram_model[1])
        self.write_to_file({"weights":weights},"i")
        self.write_to_file(self.i_probabilities,"i")

    def get_tokenized_corpus(self,text):
        tokenized_corpus=self.tokenizer.fullTokenize(text.lower())
        cleaned_tokenized_corpus=[]
        for li in tokenized_corpus:
            new_li=[]
            for i in range(2):
                new_li.insert(0,'<SOS>')
            for word in li:
                new_word=''.join(char for char in word if char not in self.punctuations)
                if(new_word!=''):
                    new_li.append(new_word)
            new_li.append('<EOS>')
            if(len(new_li)>3):
                cleaned_tokenized_corpus.append(new_li)
        return cleaned_tokenized_corpus
    
    def generate_n_gram_model(self, n, k,corpus_path):
        if(self.train_set is None or self.test_set is None):
            fp = open(corpus_path, 'r')
            text = fp.read()
            text = text.replace("\n", " ").lower()
            # ngram_freq={}
            tokenized_corpus=self.get_tokenized_corpus(text)
            test_size = 1000 if len(tokenized_corpus)>3000 else 1
            self.train_set, self.test_set = train_test_split(tokenized_corpus, test_size=test_size, random_state=1)

        n_gram_model = {}
        for word_list in self.train_set:
            if(len(word_list)==0):
                continue
            self.vocab.update(word_list)
            for i in range(len(word_list)-n+1):
                if(n==1):
                    context=""
                else:
                    context=tuple(word_list[i:i+n-1])
                next_word=word_list[i+n-1]
                if(context not in n_gram_model):
                    n_gram_model[context]={}
                n_gram_model[context][next_word]=n_gram_model[context].get(next_word,0)+1
        return n_gram_model
    
    def generate_initial_probabilities(self,n_gram_model):
        probabilities={}
        for context in n_gram_model:
            probabilities[context]={}
            tot=sum(n_gram_model[context].values())
            for word in n_gram_model[context]:                    
                probabilities[context][word]=n_gram_model[context][word]/tot
        return probabilities
    

    def load_gt_probabilities(self):
            file_path="gt"+"_probabilities_"+''+self.corpus_path.split('./')[1]
            ngram_models={}
            with open(file_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                line = line.strip()

                if line.startswith('<UNK>'):
                    # Handle <UNK> line
                    _, prob_str = line.split(':')
                    ngram_models['<UNK>'] = float(prob_str.strip())
                elif line.startswith("'"):
                    # Handle n-gram lines
                    ngram, probs_str = line.split(':', 1)
                    ngram = eval(ngram.strip())
                    probs = eval(probs_str.strip())
                    ngram_models[ngram] = probs
            self.gt_probabilities=ngram_models
            if(len(self.vocab)==0):
                for key in self.gt_probabilities:
                    if(key !='<UNK>'):
                        self.vocab.update(self.gt_probabilities[key].keys())

    def load_i_probabilities(self):
        file_path="i"+"_probabilities_"+''+self.corpus_path.split('./')[1]
        with open(file_path, 'r') as file:
            lines = file.readlines()

        weights_line = lines[0].strip()
        weights_str = weights_line.split('[')[1].split(']')[0]
        weights = [float(val) for val in weights_str.split(',')]

        n_n_gram_model = {}

        current_n_gram_model = None
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("weights"):
                continue 
            else:
                current_n_gram_model = int(line.split(":")[0].strip())
                n_n_gram_model[current_n_gram_model] = {}
                
                ngram, rest_of_line = line.split(":", 1)
                if ngram:
                    ngram = eval(ngram.strip()) 
                    probs = eval(rest_of_line.strip())
                    n_n_gram_model[current_n_gram_model] = probs
        self.i_weights=weights
        self.i_probabilities=n_n_gram_model
        if(len(self.vocab)==0):
            self.vocab.update(n_n_gram_model[1][("")].keys())


    def write_to_file(self,probabilities,type):
        file_name=type+"_probabilities_"+''+self.corpus_path.split('./')[1]
        with open(file_name, 'a') as file:
            for key, value in probabilities.items():
                file.write(f'{str(key).strip("()")}:{value}\n')

    def good_turing(self,n_gram_model):
        count_of_counts={}
        gt_probabilities={}
        total_cnt=0
        for context in n_gram_model:
            for word in n_gram_model[context]:
                curr_count=n_gram_model[context][word]
                count_of_counts[curr_count]=count_of_counts.get(curr_count,0)+1
                total_cnt+=1
        gt_probabilities["<UNK>"]=count_of_counts[1]/total_cnt


        #calculate Z and find the coefficients of linear regression.
        sorted_counts=sorted(count_of_counts.keys())
        Z=self.calculateZ(sorted_counts,count_of_counts)
        log_r=array([log(x) for x in Z.keys()])
        log_Z=array([log(Z[x]) for x in Z.keys()])
        reg = LinearRegression().fit(log_r.reshape(-1,1),log_Z)
        a=reg.coef_[0]
        b=reg.intercept_
        r_smoothed={}
        use_Zr=False


        #calculating smoothed counts using formula r_smoothed=(r+1)*S(r+1)/S(r)
        for r in sorted_counts:
            Zr = float(r+1) * (r+1)**b/ r**b  
            if (r+1) not in count_of_counts and not use_Zr:
                use_Zr=True
            if not use_Zr:
                x = (float(r+1) * count_of_counts[r+1]) / count_of_counts[r]
                Nr1=float(count_of_counts[r+1])
                Nr=float(count_of_counts[r])

                ## if abx(x-Zr) < (r+1)^2 * (N(r+1)/N(r)) *(1+(N(r+1))/N(r)) then switch to using Zr
                t=1.65*sqrt(float(r+1)**2 * (Nr1 / Nr**2)* (1. + (Nr1 / Nr)))
                if(abs(x-Zr)<t):
                    use_Zr=True
                else:
                    r_smoothed[r] = x
            if use_Zr:
                r_smoothed[r]=Zr

        for context in n_gram_model:
            curr_context_cnts=n_gram_model[context].values()
            tot=0
            for i in curr_context_cnts:
                tot+=r_smoothed[i]
            gt_probabilities[context]={}
            for word in n_gram_model[context]:
                gt_probabilities[context][word]=r_smoothed[n_gram_model[context][word]]/tot
        self.gt_probabilities=gt_probabilities
        self.write_to_file(self.gt_probabilities,"gt")


    #calculating Z values using formula Z=2*Nr/(k-i)
    def calculateZ(self,sorted_counts,count_of_counts):
        Z={}
        for (jIdx, j) in enumerate(sorted_counts):
            if jIdx == 0:
                i = 0
            else:
                i = sorted_counts[jIdx-1]
            if jIdx == len(sorted_counts)-1:
                k = 2*j - i
            else:
                k = sorted_counts[jIdx+1]
            Z[j] = 2*count_of_counts[j] / float(k-i)
        return Z
    
    def estimate_gt_probability(self,text,n=3):
        if(type(text)!=list):
            word_list = self.get_tokenized_corpus(text)[0]
        else:
            word_list=text
        prob=0
        for i in range(len(word_list)-n+1):
            context=tuple(word_list[i:i+n-1])
            next_word=word_list[i+n-1]
            if(context not in self.gt_probabilities or next_word not in self.gt_probabilities[context]):
                prob+=log(self.gt_probabilities["<UNK>"])
            else:
                prob+=log(self.gt_probabilities[context][next_word])
        return exp(prob)
    
    def estimate_probability(self,text,n=3):
        if(type(text)!=list):
            word_list = self.get_tokenized_corpus(text)[0]
        else:
            word_list=text
        for i in range(len(word_list)-n+1):
            w1 = word_list[i]
            w2 = word_list[i+1]
            w3 = word_list[i+2]
            if (w1, w2) in self.i_probabilities[3] and w3 in self.i_probabilities[3][(w1, w2)]:
                c_prob = self.i_weights[2] * self.i_probabilities[3][(w1, w2)][w3]
            else :
                c_prob=exp(-35)
            return c_prob

    def estimate_i_probability(self,text,n=3):
        if(type(text)!=list):
            word_list = self.get_tokenized_corpus(text)[0]
        else:
            word_list=text
        prob = 0
        
        for i in range(len(word_list)-n+1):
            w1 = word_list[i]
            w2 = word_list[i+1]
            w3 = word_list[i+2]
            
            c_prob = 0
            
            if (w1, w2) in self.i_probabilities[3] and w3 in self.i_probabilities[3][(w1, w2)]:
                c_prob += self.i_weights[2] * self.i_probabilities[3][(w1, w2)][w3]
            
            if (w2,) in self.i_probabilities[2] and w3 in self.i_probabilities[2][(w2,)]:
                c_prob += self.i_weights[1] * self.i_probabilities[2][(w2,)][w3]
            
            if w3 in self.i_probabilities[1][""]:
                c_prob += self.i_weights[0] * self.i_probabilities[1][""][w3]
            if(c_prob==0):
                c_prob=exp(-5)
            
            if c_prob > 0:
                prob += log(c_prob)
        
        
        return exp(prob)
    
    def calculate_perplexities(self):
        train_file_name="2023201059_LM4"+"_train-perplexity.txt"
        test_file_name="2023201059_LM4"+"_test-perplexity.txt"
        with open(test_file_name, 'w') as file:
            perplexities = []
            for word_list in self.test_set:
                if len(word_list) > 0:
                    n = len(word_list)

                    curr_prob = self.estimate_i_probability(word_list)
                    if(curr_prob == 0):
                        continue
                    perplexity=curr_prob**(-1/n)
                    sentence=' '.join(word_list[2:-1])
                    file.write(f"{sentence}\t{perplexity}\n")
                    if(perplexities!=0):
                        perplexities.append(log(perplexity))  

            avg_perplexity = exp(sum(perplexities) / len(perplexities))
            print("Average Perplexity:", avg_perplexity)
            
    def generate_next_word_gt(self,sentence,k):
        word_list=self.get_tokenized_corpus(sentence)[0]
        word_list.pop()
        context=word_list[-(self.n-1):]
        curr_words={}
        if(tuple(context) in self.gt_probabilities):
            curr_words=self.gt_probabilities[tuple(context)]

        if(len(curr_words)<k):
            for word in self.vocab:
                if(word not in curr_words):
                    possible_word_list=context+[word]
                    curr_words[word]=self.estimate_gt_probability(possible_word_list)
                if(len(curr_words)==k):
                    break
        return curr_words
            
    def generate_next_word_i(self,sentence,k):
        word_list=self.get_tokenized_corpus(sentence)[0]
        word_list.pop()
        context=word_list[-(self.n-1):]
        curr_words={}
        for word in self.vocab:
            if(word not in curr_words):
                possible_word_list=context+[word]
                curr_words[word]=self.estimate_i_probability(possible_word_list)
        curr_words = sorted(curr_words.items(), key=lambda x: x[1], reverse=True)
        return curr_words[0:k]
    

    def generate_next_word_no_smoothing(self,sentence,k):
        word_list=self.get_tokenized_corpus(sentence)[0]
        word_list.pop()
        context=word_list[-(self.n-1):]
        curr_words={}
        for word in self.vocab:
            if(word not in curr_words):
                possible_word_list=context+[word]
                curr_words[word]=self.estimate_probability(possible_word_list)
        curr_words = sorted(curr_words.items(), key=lambda x: x[1], reverse=True)
        return curr_words[0:k]


if __name__ == '__main__':
    print(sys.argv[0], sys.argv[1], sys.argv[2])
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    lm=lm(corpus_path)

    if(lm_type=="tr"):
        lm.train(corpus_path)
        lm.calculate_perplexities()


    elif(lm_type=="i"):
        lm.load_i_probabilities()
        while True:
            text = input("Enter a sentence (type 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            print("probability with linear interpolation: ",lm.estimate_i_probability(text))

    elif(lm_type=="g"):
        lm.load_gt_probabilities()
        while True:
            text = input("Enter a sentence (type 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            print("probaility with good turing smoothing: ",lm.estimate_gt_probability(text))
