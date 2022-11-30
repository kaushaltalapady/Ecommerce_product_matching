from .utils import jacard_similarity,load_vector,compute_fuzz_ratio,cosine_similarity
import pickle 
from nltk.corpus import stopwords
import numpy as np



class TextBasedMatching:
    """  
    Computes similarity based on the text and title
    """
    def __init__(self,df):
        self.df=df
        self.vectors=pickle.load(open("Product_matching\glove_vectors",'rb'))
        self.stopwords=set(stopwords.words("english"))
        self.remove_stopwords=lambda x:True if x  not in self.stopwords else False
        self.clean_df("product_name")
        self.clean_df("description")
        # self.product_vectors=list(map(load_vector,self.df['product_name'],[self.vectors]*len(self.df)))
        # self.descrip_vectors=list(map(load_vector,self.df['description'],[self.vectors]*len(self.df)))
        
    
    def clean_df_by_removing_stopwords(self,text):
        return list(filter(self.remove_stopwords,text))
    
    def merge_text(self,text):
        return ' '.join(text)
    
    def clean_df(self,column):
        self.df[column] = self.df[column].apply(str.split)
        self.df[column]= self.df[column].apply(self.clean_df_by_removing_stopwords)
        self.df[column]=self.df[column].apply(self.merge_text)    
    
    
    def compute_jacard_similarity(self,title,description):
        jacard_title_similarities=list(map(jacard_similarity,[title]*len(self.df['product_name']),self.df['product_name']))
        jacard_title_similarities=np.array(jacard_title_similarities)
        jacard_description_similarities=list(map(jacard_similarity,[description]*len(self.df['description']),self.df['description']))
        jacard_description_similarities=np.array(jacard_description_similarities)
        return (jacard_title_similarities+jacard_description_similarities)/2
    
    def compute_glove_similarity(self,title,description):
        title_vector=load_vector(title,self.vectors)
        description_vector=load_vector(description,self.vectors)
        title_similarity=np.array(list(map(cosine_similarity,[title_vector]*len(self.df),self.df['product_name'])))
        description_similarity=np.array(list(map(cosine_similarity,[description_vector]*len(self.df),self.df['description'])))
        merged_similarity=title_similarity+description_similarity
        return merged_similarity/2
    
    def compute_fuzzy_similarity(self,title,description):
        title_similarity=np.array(list(map(compute_fuzz_ratio,[title]*len(self.df),self.df['product_name'])))
        description_similarity=np.array(list(map(compute_fuzz_ratio,[description]*len(self.df),self.df['description'])))
        merged_similarity=title_similarity+description_similarity
        return merged_similarity/2
    
    def compute_text_similarity(self,title,description):
        title=' '.join(list(filter(self.remove_stopwords,title.split())))
        description=' '.join(list(filter(self.remove_stopwords,description.split())))
        jacard_similarities=self.compute_jacard_similarity(title,description)
        # glove_similarities=self.compute_glove_similarity(title,description)
        levestine_similarity=self.compute_fuzzy_similarity(title,description)
        composite_similarity=(jacard_similarities+levestine_similarity)/2
        result_dict={}
        result_dict["jacard"]={"similarities":jacard_similarities[np.argsort(jacard_similarities)[::-1][:10]],"indicies":np.argsort(jacard_similarities)[::-1][:10],"products":list(self.df['product_name'].iloc[np.argsort(jacard_similarities)[::-1][:10]])}
        # result_dict["glove"]={"similarities":glove_similarities[np.argsort(glove_similarities)[::-1][:10]],"indicies":np.argsort(glove_similarities)[::-1][:10],"products":self.df['product_name'].iloc[np.argsort(glove_similarities)[::-1][:10]]}
        result_dict["levestine"]={"similarities":levestine_similarity[np.argsort(levestine_similarity)[::-1][:10]],"indicies":np.argsort(levestine_similarity)[::-1][:10],"products":list(self.df['product_name'].iloc[np.argsort(levestine_similarity)[::-1]][:10])}
        result_dict["composite"]={"similarities":composite_similarity[np.argsort(composite_similarity)[::-1][:10]],"indicies":np.argsort(composite_similarity)[::-1][:10],"products":list(self.df['product_name'].iloc[np.argsort(composite_similarity)[::-1][:10]])}
        return result_dict

        
        
    
    
    
        
        