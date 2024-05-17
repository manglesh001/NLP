##Neural POS Tagger

#Feed Forward Neural Network POS Tagging
In the sentence "An apple a day keeps the doctor away", to
1get the POS tag for the word "a", a network with p = 2 and s = 3 would take
the embeddings for ["An", "apple", "a", "day", "keeps", "the"] and output
the POS tag "DET".
pos_tagger.py: Runs the POS tagger (command line arguments
-f for FFN, -r for RNN), which should prompt for a sentence and
output its POS tags in the specified format.
python pos_tagger.py -f
> An apple a day
an DET
apple NOUN
a DET
day NOUN

#Recurrent Neural Network POS Tagger
In the sentence "An apple a day keeps the doctor away", the
model takes the embeddings for ["An", "apple", "a", "day", "keeps", "the",
"doctor", "away"] and outputs the POS tags for all the words in the sentence
["DET", "NOUN", "DET", "NOUN", "VERB", "DET", "NOUN", "ADV"]
python pos_tagger.py -r
> An apple a day keeps the doctor away
an DET
apple NOUN
a DET
day NOUN
keeps VERB
the DET
doctor NOUN
away ADV

##Pretrained model 
In the Model Folder







