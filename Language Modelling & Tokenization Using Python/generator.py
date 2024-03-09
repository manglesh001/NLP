import sys
from language_model import lm
# class generator:
#     def generate(self):

if __name__ == '__main__':
    print(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
    lm_type=sys.argv[1]
    corpus_path=sys.argv[2]
    k=int(sys.argv[3])
    lm=lm(corpus_path)
    probs={}
    if(lm_type == 'none'):
        lm.load_i_probabilities()
        while True:
            sentence=input("Enter a sentence to generate or enter exit: ")
            if(sentence=='exit'):
                break
            print(lm.generate_next_word_no_smoothing(sentence,k))
    elif(lm_type=='i'):
        lm.load_i_probabilities()
        while True:
            sentence=input("Enter a sentence to generate or enter exit: ")
            if(sentence=='exit'):
                break
            print(lm.generate_next_word_i(sentence,k))
    elif(lm_type=="g"):
        probs=lm.load_gt_probabilities()
        while True:
            sentence=input("Enter a sentence to generate or enter exit: ")
            if(sentence=='exit'):
                break
            print(lm.generate_next_word_gt(sentence,k))
    else:
        print("Enter Correct Type")
