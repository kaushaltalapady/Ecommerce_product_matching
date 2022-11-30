import io
import numpy as np
from fuzzywuzzy import fuzz
from numpy.linalg import norm 
import regex

def load_vector(text,vectors):
  text=text.split()
  word_vector=np.zeros((300,))
  text_count=0
  for word in text:
    if word in set(vectors.keys()):
      word_vector=word_vector+vectors[word]
      text_count+=1
  if text_count:
    word_vector=word_vector/len(text)
  return word_vector

def jacard_similarity(text1,text2):
    text1=list(map(str.lower,text1.split()))
    text2=list(map(str.lower,text2.split()))
    text1=set(text1)
    text2=set(text2)
    intersection=text1.intersection(text2)
    union=text1.union(text2)
    return float(len(intersection)/len(union))

def compute_fuzz_ratio(text1,text2):
     return fuzz.ratio(text1,text2)

def cosine_similarity(vec1,vec2):
  cosine_similarity = np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
  return cosine_similarity 

def replace_regex(text):    
    return regex.sub("=>",":",text)