# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:57:37 2019

@author: Windows 10
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.spatial import distance
import math
from sklearn.metrics import silhouette_score




#%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))





c1 = pd.read_csv("Mall_Customers.csv", names=['x0', 'x1',"x2","x3","x4"], sep=";")
c1

plt.scatter(c1['x3'],c1['x4'])

#print(c1["x3"])




def complete_distance(clusters ,cluster_num):
    print('first cluster | ','second cluster | ', 'distance')
    while len(clusters) is not cluster_num:
        # Clustering           (
        closest_distance=clust_1=clust_2 = math.inf
        # for every cluster (until second last element)
        for cluster_id, cluster in enumerate(clusters[:len(clusters)]): #associates ID to each cluster in order
            for cluster2_id, cluster2 in enumerate(clusters[(cluster_id+1):]):  #loops from ID=1
                furthest_cluster_dist = -1 
# this is different from the complete link in that we try to minimize the MAX distance
# between CLUSTERS
                # go through every point in this prospective cluster as well
                # for each point in each cluster
                for point_id,point in enumerate(cluster): 
                    for point2_id, point2 in enumerate(cluster2):
# make sure that our furthest distance holds the maximum distance betweeen the clusters at focus
                        if furthest_cluster_dist < distance.euclidean(point,point2): 
                            furthest_cluster_dist = distance.euclidean(point,point2)
# We are now trying to minimize THAT furthest dist
                if furthest_cluster_dist < closest_distance:
                    closest_distance = furthest_cluster_dist
                    clust_1 = cluster_id
                    clust_2 = cluster2_id+cluster_id+1
               # extend just appends the contents to the list without flattening it out
        print(clust_1,' | ',clust_2, ' | ',closest_distance)
        clusters[clust_1].extend(clusters[clust_2]) 
        # don't need this index anymore, and we have just clustered once more
        clusters.pop(clust_2) 
    return(clusters)



def hierarchical(data, cluster_num, metric = 'complete'):
    # initialization of clusters at first (every point is a cluster)
    init_clusters=[]
    for index, row in data.iterrows():
        init_clusters.append([[row['x3'], row['x4']]])
    if metric is 'complete':
        return complete_distance(init_clusters, cluster_num)
    

    
clusters = hierarchical(c1,5)
colors = ['green', 'purple', 'teal', 'red', "brown", "yellow"]

plt.figure(figsize=(12, 12))  
plt.xlabel("Income")
plt.ylabel("Score")
         
for cluster_index, cluster in enumerate(clusters):
    for point_index, point in enumerate(cluster):
        plt.plot([point[0]], [point[1]], marker='o', markersize=6, color=colors[cluster_index])   
     
 #silhouette       
preds=[]
for cluster_index, cluster in enumerate(clusters):
    for point_index, point in enumerate(cluster):
        preds.append([cluster_index])
preds


X=c1.as_matrix()
X=X[:,(3,4)]


clusters2=clusters[0]+clusters[1]+clusters[2]+clusters[3]+clusters[4]




silhouette_score (clusters2, preds, metric='euclidean')

range_cluster = list (range(2,9))
print ("cluster from 2 to 9: \n", range_cluster)
for clust_num in range_cluster:
    
    clusters = hierarchical(c1,clust_num)
    preds=[]
    for cluster_index, cluster in enumerate(clusters):
       for point_index, point in enumerate(cluster):
           preds.append([cluster_index])
    
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For clust num = {}, silhouette score is {})".format(clust_num, score))



#print(init_clusters)
        
        
# generate the linkage matrix

X=c1.as_matrix()
X=X[:,(3,4)]



complete_link = linkage(X, 'complete',metric='euclidean')
complete_link

plt.figure(figsize=(12,12))
plt.title('Hierarchical Clustering Dendrogram')

plt.ylabel('distance')
dendrogram(
    complete_link,
    leaf_rotation=90.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
    color_threshold=70
)
plt.show()











