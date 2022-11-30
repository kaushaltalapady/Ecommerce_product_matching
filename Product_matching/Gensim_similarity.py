from gensim import corpora, models, similarities
import spacy

class Similarity:
    def __init__(self,document):
        try:
            self.document=document
            self.dictionary=None
            self.corpus=None
            self.corp_func=corpora
            self.preprocess_text()
        except:
            pass
        
    
    def preprocess_text(self):
        try:
            self.document=[[text.lower() for text in doc.split()]for doc in self.document]
            self.dictionary = self.corp_func.Dictionary(self.document)
            self.corpus=[self.dictionary.doc2bow(text) for text in self.document]
        except:
            pass
    
    def compute_similarity(self):
        raise NotImplementedError

class LSI_similarity(Similarity):
    def __init__(self,document):
        try:
            super().__init__(document)
            self.model=models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=2)
        except:
            pass
    
    def compute_similarity(self,text):
        try:
            vec_bow = self.dictionary.doc2bow(text.lower().split())
            vec_lsi = self.model[vec_bow]
            index = similarities.MatrixSimilarity(self.model[self.corpus])
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            return sims
        except:
            pass

class TFIDF_similarity(Similarity):
    def __init__(self,document):
        try:
            super().__init__(document)
            self.model=models.TfidfModel(self.corpus, id2word=self.dictionary)
        except:
            pass
    
    def compute_similarity(self,text):
        try:
            vec_bow = self.dictionary.doc2bow(text.lower().split())
            vec_lsi = self.model[vec_bow]
            index = similarities.MatrixSimilarity(self.model[self.corpus])
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            return sims
        except:
            pass


class LDA_similarity(Similarity):
    def __init__(self,document):
        try:
            super().__init__(document)
            self.model=models.LdaModel(self.corpus, id2word=self.dictionary)
        except:
            pass
        
    def compute_similarity(self,text):
        try:
            vec_bow = self.dictionary.doc2bow(text.lower().split())
            vec_lsi = self.model[vec_bow]
            index = similarities.MatrixSimilarity(self.model[self.corpus])
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            return sims
        except:
            pass
        


class Spacy_similarity:
    def __init__(self,document):
        try:
            self.model=spacy.load("en_core_web_sm")
            self.document=document
        except:
            pass
    
    def single_simlarity(self,doc1,doc2):
        try:
            doc1=self.model(doc1)
            doc2=self.model(doc2)
            return doc1.similarity(doc2)
        except:
            pass
        
    def compute_similarity(self,document):
        try:
            same_doc_list=[document]*len(self.document)
            doc_score=list(map(self.single_simlarity,same_doc_list,self.document))
            sims=sorted(enumerate(doc_score), key=lambda item: -item[1])
            return sims
        except:
            pass




class GensimSimilarity:
    def __init__(self,titles,similarity="TFIDF"):
        if similarity == "TFIDF":
            self.model=TFIDF_similarity(titles)
        elif similarity =="LDA":
            self.model=LDA_similarity(titles)
        elif similarity == "LSI":
            self.model=LSI_similarity(titles)
        self.titles=titles
    
    def compute_similarity(self,text):
        sim_ranked=self.model.compute_similarity(text)
        most_sim=sim_ranked[0][0]
        return most_sim
    
    
    
        