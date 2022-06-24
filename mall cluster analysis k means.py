# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:26:57 2019

@author: Windows 10
"""





import pandas
colnames = ["CustomerID", "Gender", "Age", "Annual_income", "Spending_score"]
data = pandas.read_csv("Mall_customers.csv", names=colnames, sep=";")


income=data.Annual_income.tolist()
score=data.Spending_score.tolist()

from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x': income,
    'y': score
})

np.random.seed(178)


#ELBOW METHOD
from sklearn.cluster import KMeans

sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');



#SILHOUETTE METHOD



#Use silhouette score
range_n_clusters = list (range(2,10))
print ("Number of clusters from 2 to 9: \n", range_n_clusters)


for n_clusters in range_n_clusters:
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(df)
    centers = clusterer.cluster_centers_

    score = silhouette_score (df, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


clusterer = KMeans (n_clusters=5)
preds = clusterer.fit_predict(df)
silhouette_score (df, preds, metric='euclidean')


clusterer.predict(df)

k = 5

# centroids[i] = [x, y]
centroids = {
    i+1: [np.random.randint(15, 137), np.random.randint(1,99)]
    for i in range(k)
}

   
fig = plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b',4:"y",5:"m",6:"c"}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.xlabel("Income")
plt.ylabel("Score")

plt.show()



def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())


fig = plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.xlabel("Income")
plt.ylabel("Score")

plt.show()



## Update Stage

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.xlabel("Income")
plt.ylabel("Score")

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

## Repeat Assigment Stage

df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.xlabel("Income")
plt.ylabel("Score")

plt.show()

# Continue until all assigned categories don't change any more
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(10, 10))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.xlabel("Income")
plt.ylabel("Score")

plt.show()



df = pd.DataFrame({
    'x': income,
    'y': score
})

        
kmeans = KMeans(n_clusters=5)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(10, 10))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(10, 140)
plt.ylim(0, 100)
plt.show()




