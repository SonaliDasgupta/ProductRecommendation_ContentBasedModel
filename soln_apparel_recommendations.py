# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:02:27 2018

@author: Admin
"""


import numpy as np
from IPython.display import display, SVG, Math, YouTubeVideo


image_features = {}

for i in range(len(asins)):
  for j, doc in data.iterrows():
    if(doc['asin'] == asins[i]):
      image_features[j] = bottleneck_features_train[i]

imageFeatures = np.array([image_features[key] for key in image_features.keys()])



def idf_w2_brand_images(doc_id, w_title, w_brand_col, w_images, num_results):
  idf_w2v_dist  = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1,-1))
  ex_feat_dist = pairwise_distances(extra_features, extra_features[doc_id])
  
  image_dist = pairwise_distances(imageFeatures, imageFeatures[doc_id].reshape(1,-1))

  pairwise_dist   = (w_title * idf_w2v_dist +  w_brand_col * ex_feat_dist + w_images * image_dist)/float(w_title + w_brand_col + w_images)
  
  
  indices = np.argsort(pairwise_dist.flatten())[0:num_results]
  #pdists will store the num_results smallest distances
  pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

  #data frame indices of the num_results smallest distances
  df_indices = list(data.index[indices])
    

  for i in range(len(indices)):
    heat_map_w2v_brand(data['title'].loc[df_indices[0]],data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], indices[0], indices[i],df_indices[0], df_indices[i], 'weighted')
    print('Product Title: ', data['title'].loc[df_indices[i]])
           
    print('Euclidean Distance from input image:', pdists[i])
    print('Amazon Url: www.amzon.com/dp/'+ asins[indices[i]])  
            
    print('ASIN :',data['asin'].loc[df_indices[i]])
    print('Brand :',data['brand'].loc[df_indices[i]])
    print('euclidean distance from input :', pdists[i])
    print('='*125)
            
            
idf_w2_brand_images(12566, 1,1,1, 20)
      