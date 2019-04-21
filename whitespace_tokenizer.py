from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import spacy
import re
# import nltk

stop_words =['ours', 'keep', 'in', 'enough', 'anything', 'latterly'
 , 'thereupon', 'your', 'if', 'as', 'each', 'his', 'but'
 , 'everywhere', 'hereupon', 'being', 'becoming', 'and',
 'anyhow', 'serious', 'something', 'latter', 
 'namely', 'name', 'seemed', 'yourselves', 'toward', 'must', 
 'same', 'then', 'become', 'while', 'becomes', 'ourselves', 'perhaps', 
 'or', 'more', 'whose', 'along', 'own', 'thence', 'had', 'itself', 
 'top', 'whether', 'beside', 'into', 'on', 'per', 'whole', 'one', 
 'towards', 'himself', 'against', 'beyond', 'off', 'done', 'are', 
 'you', 'he', 'yours', 'an', 'myself', 'themselves', 
 'hereafter', 'else', 'have', 'neither', 'again', 'afterwards', 
 'under', 'its', 'due', 'always', 'be', 'over', 'therefore', 
 'very', 'at', 'during', 'nobody', 'where', 
 'whoever', 'across', 'thereafter', 'i', 'thereby', 'empty', 
 'move', 'put', 'through', 'since', 'my', 'wherein', 'became', 'thus',
 'none', 'cannot', 'did', 'next', 'above', 'regarding', 
 'to', 'too', 'within', 'just', 'nothing', 'now', 'am', 'part', 'seems', 'than', 'alone', 'after', 'once', 
 'doing', 'otherwise', 'who', 'indeed', 'full', 'whence',
 'before', 'how', 'although', 'mostly', 'take', 'between', 'these',
 'whereas', 'former', 'whom', 'many', 'amongst', 'other',
 'ca', 'besides', 'go', 'much', 'may', 'nowhere', 'together', 
 'him', 'her', 'there', 'say', 'throughout',
 'whereby', 'mine', 'formerly', 'only', 'really', 'herein', 
 'show', 'might', 'hers', 'often', 'when', 
 'whereupon', 'those', 'rather', 'somewhere', 'give', 'here', 
 'do', 'used', 'does', 'me', 'seem', 'unless', 'sometime', 
 'almost', 'via', 'back', 'hereby', 'few', 'all', 'up', 
 'using', 'should', 'well', 'see', 'been', 'various', 'yourself', 
 'bottom', 'onto', 'side', 'for', 'everyone', 'will', 
 'several', 'however', 'meanwhile', 'can', 'everything', 'around', 
 'she', 'of', 'their', 'were', 'get', 'until', 'that', 
 'yet', 'already', 'both', 'by', 'somehow', 'any', 'please', 
 'whereafter', 'behind', 'therein', 'the', 'they', 'whenever', 
 'out', 'still', 'our', 'most', 'least', 'though', 'with', 'a', 'could',
 'such', 'less', 'was', 'nor', 'others', 'why', 'about', 'never', 'so',
 'us', 'wherever', 'beforehand', 'moreover', 'last', 'among', 'elsewhere',
 'nevertheless', 'quite', 'upon', 'ever', 'anywhere', 'we', 'down', 'what', 
 'amount', 'whither', 'it', 'below', 'someone', 'either', 'is', 'some',
 'even', 'also', 'from', 'except', 'further', 'herself', 'make', 'which',
 'this', 'call', 'without', 'made', 're', 'sometimes', 'another', 'whatever',
 'anyone', 'would', 'every', 'thru', 'them', 'anyway', 'hence', 'has', 
 'because', 'seeming',"what's","whats",'-PRON-','iam', 'im',"i'm","what's","whats",'am']

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

class WhitespaceTokenizer(Tokenizer, Component):
    name = "tokenizer_whitespace"

    provides = ["tokens"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        # spacy_nlp = spacy.load('en_core_web_sm')
        # spacy_nlp.Defaults.stop_words |= {"no","not",""}
        # doc = spacy_nlp(text)
        # # print(words)
        # tok = [tokend for tokend in doc if not tokend.is_stop]

        # print(tok,'...............')
        # tokensd = nltk.tokenize.word_tokenize(text)
        # nltk_stopwords = nltk.corpus.stopwords.words('english')
        # tokensd = [tokend for tokend in tokensd if not tokend in nltk_stopwords]
        # print(tokensd,'----------------------')
        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        # print(t(text)
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text).split()
        tokensd = []
        # for tokend in words:
        #     if not tokend.lower() in stop_words:
        #         if len(tokend.lower())>2:
        #             tokensd.append(tokend.lower())
        #         elif tokend.isnumeric():
        #             tokensd.append(tokend.lower())
        tokensd = [tokend.lower() for tokend in words if not tokend.lower() in stop_words]
        doc = nlp(str(' '.join(tokensd)))
        words = [str(lemm.lemma_) for lemm in doc]
        words = [re.sub(r'[^\x00-\x7f]','',re.sub('[\t\r\n,)([\]!%|!#$%&*+,.-/:;<=>?@^_`{|}~?]','',str(i))).strip() for i in words]
        # print(tokensd)
        running_offset = 0
        tokens = []
        texts = ' '.join(words)
        for word in words:
            word_offset = texts.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        # print('//////////////////')
        # print(tokensd)
        # print('//////////////////')
        
        return tokens