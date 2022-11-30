import numpy as np
import ast 
from .utils import replace_regex

class SpecificationSimilarity:
    def __init__(self,df):
        self.df=df 
        self.preprocess_df()
    
    
    def preprocess_df(self):
        self.df['brand']=self.df['brand'].fillna('no brand')
        self.df['brand']=self.df['brand'].apply(str.lower)
        self.df['product_category_tree']=self.df['product_category_tree'].fillna("['']")
        self.df["product_specifications"]=self.df["product_specifications"].fillna('{"product_specification":[]}')
        self.df['product_category_tree']=self.df['product_category_tree'].apply(self.split_product_category_tree)
        self.df['product_specifications']=self.df['product_specifications'].apply(replace_regex)
        self.df['product_specifications']=self.df['product_specifications'].apply(self.convert_to_dict_format)
        
    
    def split_product_category_tree(self,product_tree):
        product_tree=ast.literal_eval(product_tree)
        product_tree=product_tree[0].split(">>")
        product_tree=list(map(str.strip,product_tree))
        product_tree=list(map(str.lower,product_tree))
        return product_tree
    
    
    def convert_to_dict_format(self,product_spec):
        try:
            product_spec = ast.literal_eval(product_spec)
        except:
            return ''
        # print(product_spec)
        product_spec=product_spec['product_specification']
        product_spec_dict={}
        for prod_dict in product_spec:
            try:
                product_spec_dict[prod_dict['key'].lower()]=prod_dict['value'].lower()
            except:
                pass 
        return product_spec_dict
    
    
        
    def compute_simlarity(self,product_specs):
        brand_score =self.check_brand(product_specs['brand'])
        product_tree_score=self.check_product_tree(product_specs['product_category_tree'])
        product_spec_score=self.check_product_specs(product_specs["product_specifications"])
        composite_score=(0.2*brand_score+0.4*product_tree_score+0.4*product_spec_score)
        composite_locs=np.argsort(composite_score)[::-1][:10]
        return {"similarities":composite_score[composite_locs],"indicies":composite_locs,"titles":list(self.df['product_name'].iloc[composite_locs])}
        
        
        
    
    def check_brand(self,brand):
        return np.array(list(map(self.same_brand,self.df['brand'],len(self.df)*[brand])),dtype=float) 
    
    def same_brand(self,brand1,brand2):
        if brand1==brand2:
            return 1
        else:
            return 0
    
    def check_product_tree(self,product_tree):
        return np.array(list(map(self.score_product_tree,[product_tree]*len(self.df),self.df['product_category_tree'])),dtype=float)
    
    def score_product_tree(self,test_tree,data_tree):
        test_tree=set(test_tree)
        data_tree=set(data_tree)
        intersection =test_tree.intersection(data_tree)
        return len(intersection)/len(test_tree)
    
    def check_product_specs(self,test_spec):
        return np.array(list(map(self.score_the_specs,[test_spec]*len(self.df),self.df['product_specifications'])),dtype=float)
    
    def score_the_specs(self,test_spec,data_spec):
        test_spec_count=len(test_spec.keys())
        spec_count=0
        for key in test_spec.keys():
            if type(data_spec)!=dict:
                break
            if key in data_spec.keys():
                if test_spec[key]==data_spec[key]:
                    spec_count+=1
        
        return spec_count/test_spec_count
    
    
    
        
    
    
    