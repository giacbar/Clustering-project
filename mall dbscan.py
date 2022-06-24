# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:50:00 2019

@author: Windows 10
"""




import pandas as pd
import numpy as np
import scipy.spatial
import sklearn
from sklearn.preprocessing import StandardScaler
import scipy as sci
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score





colnames = ["CustomerID", "Gender", "Age", "Annual_income", "Spending_score"]
data = pd.read_csv("Mall_customers.csv", names=colnames, sep=";")
income=data.Annual_income.tolist()
score=data.Spending_score.tolist()
df = pd.DataFrame({
    'x': income,
    'y': score
})
#X=df


mall = pd.read_csv('Mall_customers.csv', sep=';', header=None)

n,q = mall.shape
c = list(mall.columns.values)
p = q-1
X = mall.as_matrix(c[3:5])
X


# Data import
# X data

def dbscan(X,min_size_cluster,dist_neighbor):
    ## Perform the DBSCAN clustering algorithm on X
    # X : data matrix (the rows represent the observations) size : n x p
    # min_size_cluster : minimum number of points in each cluster
    # dist_neighbor : radius of neighborhood
    n,_ = X.shape
    kdtree = scipy.spatial.KDTree(X)
    X_visit = np.zeros((n,1),dtype=bool)
    clusters = np.zeros((n,1))
    c = 0
    for k in range(0,n):
        if X_visit[k]==0: # if k has not been visited
            X_visit[k] = 1 # k is flagged
            Neighbors = kdtree.query_ball_point(X[k,:],dist_neighbor)
#            Neighbors = neighborhood(X,X[k,:],dist_neighbor) # we look for its neighbors
            if len(Neighbors)<min_size_cluster:
                clusters[k] = -1
            else: # If the point k has enough neighbors
                c = c+1
                clusters[k] = c # we create a cluster and k belong to it
                while len(Neighbors)>0: # Then we study all of its neighbors
                    i = Neighbors.pop()
                    if X_visit[i]==0:   # if i has not been visited, we flag him
                        X_visit[i] = 1
                        Vi = kdtree.query_ball_point(X[i,:],dist_neighbor) # and we look for its neighbors too
                        if len(Vi)>=min_size_cluster: # if it has enough neighbors, we treat them too
                            Neighbors.extend(Vi)
                    if clusters[i]<=0:
                        clusters[i] = c
            
    return clusters



#Finding the optimal distance given number of points per cluster

plt.figure(figsize=(15, 15)) 
plt.xlabel("k-distances")
plt.ylabel("eps")

for min_size_cluster in (5,10,15):
   nbrs = NearestNeighbors(n_neighbors=min_size_cluster).fit(df)
   distances, indices = nbrs.kneighbors(df)
   distanceDec = sorted(distances[:,min_size_cluster-1], reverse=True)
             
   plt.plot(indices[:,0], distanceDec)

dist_neighbor = 10
min_size_cluster = 5
output = dbscan(X,min_size_cluster,dist_neighbor)
#outputfixed=["none"]*len(output)

#for i in range(0,len(output)):
 #   if int(output[i][0])!=-1:
 #        outputfixed[i]=int(output[i][0])
         
#poutputfixed

#for i in range(0,len(output)):
 #   if outputfixed[i] == "none":
  #      del outputfixed[i]
        
#outputfixed = output()
silhouette_score (X, output, metric='euclidean')

dist_neighbor = 15
min_size_cluster = 10
output = dbscan(X,min_size_cluster,dist_neighbor)
silhouette_score (X, output, metric='euclidean')

dist_neighbor = 20
min_size_cluster = 15
output = dbscan(X,min_size_cluster,dist_neighbor)
silhouette_score (X, output, metric='euclidean')


#silhouette

range_n_min_size_clusters = list (range(2,16))
print ("min size of clusters from 2 to 16: \n", range_n_min_size_clusters)
for min_size_clusters in range_n_min_size_clusters:
    
    output = dbscan(X,min_size_clusters,dist_neighbor)
    preds = output
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For min_size_clusters = {}, silhouette score is {})".format(min_size_clusters, score))


#Running the algorithm
output = dbscan(X,min_size_cluster,dist_neighbor)
#r = np.mean(output==y)

#adding colors
vector=["none"]*len(output)
for i in range(0,len(output)):
    
        if int(output[i][0])==1:
            vector[i]="r"
        elif int(output[i][0])==2:
            vector[i]="g"
        elif int(output[i][0])==3:
            vector[i]="b"
        elif int(output[i][0])==4:
            vector[i]="y"
        else:
            vector[i]="c"   
              
#plotting the clusters
df2 = pd.DataFrame({
    'x': income,
    'y': score,
    "n": vector
})

fig = plt.figure(figsize=(10, 10))           
plt.scatter(df2["x"], df2["y"], color=df2["n"])     
plt.xlabel("Income")
plt.ylabel("Score")


#silhouette
silhouette_score (X, output, metric='euclidean')

range_n_min_size_clusters = list (range(2,16))
print ("min size of clusters from 2 to 16: \n", range_n_min_size_clusters)
for min_size_clusters in range_n_min_size_clusters:
    
    output = dbscan(X,min_size_clusters,dist_neighbor)
    preds = output
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For min_size_clusters = {}, silhouette score is {})".format(min_size_clusters, score))

range_dist = list (range(2,15))
print ("min dist from 2 to 15: \n", range_dist)
for min_dist in range_dist:
    
    output = dbscan(X,min_size_cluster,min_dist)
    preds = output
    score = silhouette_score (X, preds, metric='euclidean')
    print ("For min dist = {}, silhouette score is {})".format(min_dist, score))


    
    
    
U=X

from sklearn.cluster import DBSCAN 
db = DBSCAN(eps=5, min_samples=6).fit(U)
db.labels_

vector=["none"]*len(db.labels_)
print(vector)
for i in range(0,len(db.labels_)):
    
        if int(db.labels_[i])==1:
            vector[i]="r"
        elif int(db.labels_[i])==2:
            vector[i]="g"
        elif int(db.labels_[i])==0:
            vector[i]="m"
        elif int(db.labels_[i])==3:
            vector[i]="b"
        elif int(db.labels_[i])==4:
            vector[i]="y"
        else:
            vector[i]="c"   
            
df2 = pd.DataFrame({
    'x': income,
    'y': score,
    "n": vector
})


fig = plt.figure(figsize=(10, 10))           
plt.scatter(df2["x"], df2["y"], color=df2["n"])     
plt.xlabel("Income")
plt.ylabel("Score")













from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
U_scaled = scaler.fit_transform(X)
# cluster the data into five clusters
dbscan = DBSCAN(eps=0.3, min_samples = 5)
clusters = dbscan.fit(U_scaled)
clusters.labels_


vector=["none"]*len(db.labels_)
print(vector)
for i in range(0,len(db.labels_)):
    
        if int(db.labels_[i])==1:
            vector[i]="r"
        elif int(db.labels_[i])==2:
            vector[i]="g"
        elif int(db.labels_[i])==0:
            vector[i]="m"
        elif int(db.labels_[i])==5:
            vector[i]="y"
        elif int(db.labels_[i])==6:
            vector[i]="y"
        elif int(db.labels_[i])==3:
            vector[i]="b"
        elif int(db.labels_[i])==4:
            vector[i]="y"
        else:
            vector[i]="c"   
            
result = np.hstack((U_scaled, np.atleast_2d(vector).T)) 
# plot the cluster assignments
plt.scatter(result[:, 0], result[:, 1],color=result[:,2])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")





# cluster the data into five clusters
dbscan = DBSCAN(eps=0.3, min_samples = 5)
clusters = dbscan.fit(X)
clusters.labels_

vector=["none"]*len(db.labels_)
print(vector)
for i in range(0,len(db.labels_)):
    
        if int(db.labels_[i])==1:
            vector[i]="r"
        elif int(db.labels_[i])==2:
            vector[i]="g"
        elif int(db.labels_[i])==0:
            vector[i]="m"
        elif int(db.labels_[i])==5:
            vector[i]="y"
        elif int(db.labels_[i])==6:
            vector[i]="y"
        elif int(db.labels_[i])==3:
            vector[i]="b"
        elif int(db.labels_[i])==4:
            vector[i]="y"
        else:
            vector[i]="c"   
            

result = np.hstack((X, np.atleast_2d(vector).T)) 

# plot the cluster assignments
plt.scatter(result[:, 0], result[:, 1],color=result[:,2])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

