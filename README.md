# Ecommerce_product_matching
This project i have tried various ways of matching ecommerce products, with each other to  display price diffrence. All this done using streamlit web app \n 

We can try to to solve this problem using various approches <br />
*  Based on similarity between title and description of the diffrent eccomerce items 
*  Using gensim package 
*  Using pretrained embdeddings (image or text) similarity
*  using specs provided in the data 

## Text similarity 
The principle here is that the Similar objects have similar titles and descriptions, which is an reasonable assumption. I have used 2 ways to compute the similarity
* jacard similarity
* levestine similarity
### Jacard similarity
jacard similarity can be defined as ratio of nuumber of common words between two sentences to the total nunber of words,I have used a custom function to compute the similarity <br />
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/eaef5aa86949f49e7dc6b9c8c3dd8b233332c9e7)
### Levestine similarity 
Informally, the Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one word into the other.
I have used fuzzywuzzy implemenation for the metric.<br />
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/70962a722b0b682e398f0ee77d60c714a441c54e)
### Composite similarity
Here i basically take the average  of the previous two metric 

## Gensim similarity 
Gensim is popular NLP library which could be used to the do various things from create custom word embeddings to compute text similarity, for  our use case I have used  various from models pakage of gensim which include TFIDF,LSI and LDA

## Shoppe database pretrained BERT model embedding 
Two years ago kaggle hosted a competition where participants had to match similar objects in an eccomerce into a category,given text and image data people tried various approaches but the most popular one was model(image based or text based) follwed by an arcface layer which basically ensures that the out put vector before classification layer has a property that the objects of same class is closer compared to objects of different class  this is very useful in generating embedddings of objects in which cosine similarity can be used as the metric two know similarity between two text gien a generated vector from model
The link for the kaggle compettions is :<br/>
https://www.kaggle.com/c/shopee-product-matching

## Specification Based Matching 
Probably the simplest approach which could be used in solving the problem is given data about the product tree,product specifications(like color of product etc) and brand of product,I created a custom metric which computes similarity using all three information, thing to note here is that the all there parameters are not weighted equally product specification and product tree is weighed more than brand
#### SIMILARITY = 0.2 x BRAND_SIMILARITY+0.4 x PRODUCT_SPEC_SIMILARITY+0.4 x PRODUCT_TREE_SIMILARITY

## METHODS DEPRECATED
Some methods needed to shelved since i was bulding an web app and latency was concern  chief of them where
* Similarity using glove vector 
* Image similarity( shoppe pre trained model)
### Glove Vector based Similarity 
Glove provides word embeddings for various english words which could used to compute an embedding for the whole sentence, on which similarity based on distances like cosine similarity can be computed ( since word embeddings similar words are closer together) 
### Image similarity 
Similar to the bert model which was used to compute similarity based on text CNN can be used to generate the embeddings 

## Future Approaches 
Most of the techniques are un-supervised, but we can try to use supervised
### Simple vectorization followed by the classifier 
We can use Simple vectorizer Tfidf and label and create a dataset which could be used to train a simple linear regression
### Siamese network
A step ahead previous approach is that to feed the tests into siamese network and train it based on labeled data with loss functions like Tiplet loss, contrastive loss etc

## Web App
The Web app initially takes the Query file (file whose products has to be matched),Then We also need to update the file where we fetched the similar products from(matching file) after which we need the row to which query from  then we need to select the method to compute similarity then the web app provides results  
