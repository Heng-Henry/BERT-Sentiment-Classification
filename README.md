# BERT-Sentiment-Classification

## Outline
I choose BERT,which is a transformer-based deep learning model to judge the sentiment of the iMDB movie review into positive and negative.
For comparison of the performance, I also implemented n-gram and RNN as the baseline model.

## Training 

The two pre-training steps are MLM(Masked Language Model) 
and NSP(Next Sentence Prediction) respectively. 
### MLM: 
BERT takes in a sequence of words as input,randomly masking 
some of them. The model the learns to predict the masked 
words based on the context provided by the other words in the 
sequence. This step allows the model to learn contextualized 
representations of words,which can be useful for a wide range 
of downstream tasks. 

### NSP: 
BERT takes in two consecutive sentences as input,and learns to 
predict whether the second sentence follows the first in the 
original text or not,It allows the model to learn the 
relationships between sentences,which can be useful for tasks 
such as natural language inference and question answering. 

## THE DIFFERENCE BETWEEN BERT AND DISTILBERT 
### 1. Model size: 
BERT is a larger model whereas DistilBERT is a smaller and 
more efficient one of BERT.DistilBERT is faster to train and 
use,and more suitable for applications that require less 
computational resources. 

### 2. Training Time: 
BERT can take several days to train on multiple 
GPUs,whereas DistilBERT can be trained in a few hours on a 
single GPU. 

### 3. Accuracy: 
BERT is a SOTA result on various NLP tasks,but the cost of 
computational resources is high,so DistilBERT is a 
substitution for BERT because it has a similar accuracy and 
much shorter training time and computational resource. 

### 4. Compression Technique: 
DistilBERT uses knowledge distillation to compress 
BERT,which involves training a smaller student 
model(DistilBERT) to mimic the behavior of a larger teacher 
model by transferring its knowledge during the training 
process.it makes DistilBERT a smaller and more efficient 
model without significant loss in performance. 
