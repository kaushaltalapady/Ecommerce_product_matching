import os
import gc
import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset 
from transformers import AutoTokenizer, AutoModel
import sys
import pandas as pd
from .utils import cosine_similarity
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import gc
import timm


class CFG:
    compute_cv = True  # set False to fast save
    todo_predictions = ['predictions']
    
    ### CNN and BERT
    use_amp = True
    scale = 30  # ArcFace
    margin = 0.5  # ArcFace
    seed = 2021
    classes = 11014
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    ### BERT 1
    if 'kaggle_web_client' in sys.modules:
        bert_model_name = 'Product_matching\\models\\paraphrase xlm r multilingual v1'  # for kaggle notebook
    else:
        bert_model_name = 'Product_matching\\models\\paraphrase xlm r multilingual v1' 
    
    bert_model_path = "Product_matching\\models\\paraphrase-xlm-r-multilingual-v1_epoch7-bs16x1.pt"
    max_length = 128
    bert_batch_size = 32
    bert_fc_dim = 768
    bert_use_fc = True
    image_model_name = 'tf_efficientnet_b5_ns'
    image_model_path="Product_matching\models\arcface_512x512_eff_b5_.pt"
    

        

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.criterion = nn.CrossEntropyLoss()
                
    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if CFG.use_amp:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()  # if CFG.use_amp
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=CFG.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output, self.criterion(output,label)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class ShopeeBertModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.classes,
        model_name = None,
        fc_dim = 768,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = True        
    ):

        super(ShopeeBertModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))
        print(os.getcwd())
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        print("finished loading tokenizer")
        self.backbone = AutoModel.from_pretrained(model_name).to(CFG.device)
        print("abc")
        in_features = 768
        self.use_fc = use_fc
        
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        in_features = fc_dim
            
        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, texts, labels=torch.tensor([0])):
        features = self.extract_features(texts)
        if self.training:
            logits = self.final(features, labels.to(CFG.device))
            return logits
        else:
            return features
        
    def extract_features(self, texts):
        encoding = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=CFG.max_length, return_tensors='pt').to(CFG.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        embedding = self.backbone(input_ids, attention_mask=attention_mask)
        x = mean_pooling(embedding, attention_mask)
        
        if self.use_fc:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        
        return x


def get_bert_embeddings(df,column, model_name=CFG.bert_model_name, model_path=CFG.bert_model_path,
                                      fc_dim=CFG.bert_fc_dim, use_fc=CFG.bert_use_fc, chunk=CFG.bert_batch_size):
    
    print('Getting BERT ArcFace embeddings...')
    
    model = ShopeeBertModel(model_name=model_name, fc_dim=fc_dim, use_fc=use_fc)
    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    bert_embeddings = torch.zeros((df.shape[0], 768)).to(CFG.device)
    for i in tqdm(list(range(0, df.shape[0], chunk)) + [df.shape[0]-chunk], ncols=100):
        titles = []
        for title in df[column][i : i + chunk].values:
            try:
                title = ' ' + title.encode('utf-8').decode("unicode_escape").encode('ascii', 'ignore').decode("unicode_escape") + ' '
            except:
                pass
            title = title.lower()
            
            titles.append(title)
            
        with torch.no_grad():
            if CFG.use_amp:
                with torch.cuda.amp.autocast():
                    model_output = model(titles)
            else:
                model_output = model(titles)
            
        bert_embeddings[i : i + chunk] = model_output
    
    del model, titles, model_output
    gc.collect()
    torch.cuda.empty_cache()
    
    return bert_embeddings


class ShoppeBertSimilarity:
    def __init__(self,df,column):
        self.df=df 
        self.column = column
        self.df_embeddings=get_bert_embeddings(self.df,self.column)
    
    def get_similarities(self,data):
        test_df = pd.DataFrame({self.column:[data]})
        embeddings=get_bert_embeddings(test_df,self.column)
        embeddings=embeddings[0]
        similarities=list(map(cosine_similarity,len(self.df)*[embeddings],self.df_embeddings))
        similarities=np.array(similarities)
        sorted_indicies= np.argsort(similarities)[::-1][:10]
        similarities=similarities[sorted_indicies]
        products=list(self.df[self.column].iloc[sorted_indicies])
        return {"similarities":similarities,"indexes":sorted_indicies,"products":products}
        
class ShopeeImageDataset(Dataset):
    def __init__(self, image_paths, transforms=None):

        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       

        return image,torch.tensor(1)
    
    


class ShopeeImageModel(nn.Module):

    def __init__(
        self,
        n_classes = CFG.classes,
        model_name ='tf_efficientnet_b5_ns',
        fc_dim = 512,
        margin = CFG.margin,
        scale = CFG.scale,
        use_fc = False,
        pretrained = False):


        super(ShopeeImageModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'efficientnet_b3':
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'tf_efficientnet_b5_ns':
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif model_name == 'nfnet_f3':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        #logits = self.final(feature,label)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x

def get_test_transforms():
    
    return A.Compose(
        [
            A.Resize(CFG.img_size,CFG.img_size,always_apply=True),
            A.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )

def get_image_embeddings(image_paths, model_name = CFG.image_model_name):
    embeds = []
    model = ShopeeImageModel(model_name = model_name)
    model.eval()
    model.load_state_dict(torch.load(CFG.image_model_path))
    model = model.to(CFG.device)

    image_dataset = ShopeeImageDataset(image_paths=image_paths,transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )
    
    
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            feat = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


class ShoppeImageSimilarity:
    def __init__(self,df,column):
        self.df=df 
        self.column = column
        self.df_embeddings=get_image_embeddings(self.df[self.column])
    
    def get_similarities(self,data):
        test_df = pd.DataFrame({self.column:[data]})
        embeddings=get_image_embeddings(test_df,self.column)
        embeddings=embeddings[0]
        similarities=list(map(cosine_similarity,len(self.df)*[embeddings],self.df_embeddings))
        similarities=np.array(similarities)
        sorted_indicies= np.argsort(similarities)[::-1][:10]
        similarities=similarities[sorted_indicies]
        products=list(self.df[self.column].iloc[sorted_indicies])
        return {"similarities":similarities,"indexes":sorted_indicies,"products":products}


