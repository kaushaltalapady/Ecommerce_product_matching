import streamlit as st
import pandas as pd 
from Product_matching import TextBasedMatching,ShoppeBertSimilarity,ShoppeImageSimilarity,GensimSimilarity,SpecificationSimilarity
from io import BytesIO,StringIO
import numpy as np
import ast
import regex

def create_input_data(index,df):
    row = df.iloc[index]
    out_dict={}
    out_dict['brand']=row['brand'].lower()
    product_tree=row["product_category_tree"]
    # product_tree=regex.sub(">>",',',row["product_category_tree"])
    product_tree=ast.literal_eval(product_tree)
    product_tree=product_tree[0].split(">>")
    product_tree=list(map(str.strip,product_tree))
    product_tree=list(map(str.lower,product_tree))
    out_dict['product_category_tree']=product_tree
    product_specs=row['product_specifications']
    product_specs=regex.sub("=>",':',product_specs)
    # print(product_specs)
    product_specs=ast.literal_eval(product_specs)
    product_specs=product_specs["product_specification"]
    product_specs_dict={}
    for dict_obj in product_specs:
        try:
            product_specs_dict[dict_obj['key'].lower()]=dict_obj['value'].lower()
        except:
            pass
    out_dict['product_specifications']=product_specs_dict
    return out_dict

st.title("ECCOMERCE PRODUCT MATCHER")

# st.text("Upload the file to be queried")
st.write("### File which is queried")
query_file=st.file_uploader(":",type={"csv"})
st.write("### File to be matched")
match_file=st.file_uploader("",type={"csv"})
# query_file.seek(0)
# match_file.seek(0)
try:
    # print(1)

    query_file=pd.read_csv(query_file,encoding="ISO-8859-1")
    # print(2)
    
    match_file=pd.read_csv(match_file,encoding="ISO-8859-1")
    # print(3)
    option_selected=st.selectbox("Select the method to calculate similarity",options=["Text similarity","Pretrained text model similarity","Gensim similarity","Spec based similarity"])
    try:
        st.write("### Enter the Row Id")
        row_id=int(st.text_input(""))
    except:
        row_id=0
    if option_selected=="Text similarity":
        match_file['description']=match_file['description'].fillna('')
        second_option=st.selectbox("select metric",options=["jacard",'levestine','composite'])
        text_match_object=TextBasedMatching(match_file)
        row=query_file.iloc[row_id]
        result=text_match_object.compute_text_similarity(row['product_name'],row["description"])
        st.write("### query value")
        st.dataframe(query_file[['product_name','retail_price']].iloc[row_id])
        st.write("### match value")
        # print(result)
        out=match_file[['product_name','retail_price']].iloc[result[second_option]['indicies']]
        st.write(out)
    elif option_selected=="Pretrained text model similarity":
        bert_similarity=ShoppeBertSimilarity(match_file,"product_name")
        row=query_file.iloc[row_id]
        result=bert_similarity.get_similarities(row['product_name'])
        st.write("### query value")
        st.dataframe(query_file[['product_name','retail_price']].iloc[row_id])
        st.write("### match value")
        # print(result)
        out=match_file[['product_name','retail_price']].iloc[result['indexes']]
        st.write(out)
    elif option_selected=="Gensim similarity":
        # print('a')
        gensim_similarity=GensimSimilarity(match_file["product_name"])
        row=query_file.iloc[row_id]
        result=gensim_similarity.compute_similarity(row['product_name'])
        st.write("### query value")
        st.dataframe(query_file[['product_name','retail_price']].iloc[row_id])
        st.write("### match value")
        # print(result)
        out=match_file[['product_name','retail_price']].iloc[result]
        
        st.write(out)
    elif option_selected=="Spec based similarity":
        spec_similarity=SpecificationSimilarity(match_file)
        input_data=create_input_data(row_id,query_file)
        result=spec_similarity.compute_simlarity(input_data)
        st.write("### query value")
        st.dataframe(query_file[['product_name','retail_price']].iloc[row_id])
        st.write("### match value")
        # print(result)
        out=match_file[['product_name','retail_price']].iloc[result['indicies']]
        st.write(out)
    
except Exception as e:
    print(e)

    
    
