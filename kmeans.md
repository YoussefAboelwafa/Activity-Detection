# K-means
 K-means is a popular unsupervised machine learning algorithm used for clustering data into groups, often referred to as "clusters". The algorithm aims to partition the data points into a predefined number of clusters (k), where each data point belongs to the cluster with the nearest mean (centroid). It works iteratively by initially randomly selecting k centroids, assigning each data point to the nearest centroid, recalculating the centroids based on the mean of the points assigned to each cluster, and repeating these steps until convergence or a predefined number of iterations. K-means seeks to minimize the within-cluster sum of squares.

## K-means Pseudocode
```python
def kmeans(k, current_centroids, train_set):
    for _ in range(1000): 
        old_centroids = np.copy(current_centroids)
        distances = euclidean_distances(train_set, current_centroids)
        labels = np.argmin(distances, axis=1)+1
        current_centroids = np.array([np.mean(train_set[labels == i], axis=0) for i in range(1, k+1)])
        
        # Check for convergence
        if np.allclose(old_centroids, current_centroids):
            break
    
    return labels, current_centroids  
```
Note that I used in my implementation the **max number of iterations = 1000**

## Random Starts
- Since K-means depends on the initial choice of centroids so we don't know the centroids which gives us the best result.
- Here comes the technique of random starts, where K-means algorithm is executed **20** times and the best run which gives us the lowest **WCSS** (Within Cluster Sum Squared) from the 20 runs.

Here is how it looks like
```python
def BestRandomStart(train_set,k):
    Best_labels=np.zeros(train_set.shape[0])
    Best_centroids=np.zeros((k,train_set.shape[1]))
    min_wcss=float('inf')
    for _ in range(20):       
        initial_centroids = train_set[np.random.choice(train_set.shape[0], k, replace=False)]
        labels,centroids=kmeans(k,initial_centroids,train_set)
        lists = [[] for _ in range(k)]
        for i in range(len(labels)):
            lists[labels[i]-1].append(i)
        WCCS=0    
        for j in range(len(lists)):
            WCCS+=calculate_wcss(lists[j],centroids[j],train_set)
        if WCCS<min_wcss:
            min_wcss=WCCS
            Best_labels=labels
            Best_centroids=centroids
    return Best_labels,Best_centroids            
```
## Hyper Parameter Tuning
K-means has a hyper parameter which is (K) number of clusters.<br>
How to choose the Best K ?
### 1. Elbow Method
The Elbow Method is a heuristic technique used to determine the optimal number of clusters (k) in a k-means clustering algorithm. It involves plotting the within-cluster sum of squares (WCSS) against the number of clusters. 

When the number of clusters is increased, the WCSS typically decreases, as the clusters become more specific to the data points. However, beyond a certain point, adding more clusters does not lead to significant reductions in the WCSS, resulting in diminishing returns.

The Elbow Method suggests selecting the number of clusters at the "elbow" point of the plot, where the rate of decrease in the WCSS sharply decreases, forming an elbow-like curve. 

![alt text](<KMeans Plots/elbow_mean.png>)

![alt text](<KMeans Plots/elbow_flatten.png>)

- Sometimes the plot doesn't have an obivious elbow point as it is in the two plots above. <br>
- Is there a better method?
### 2. Silhouette Method
The Silhouette score is a very useful method to find the number of K when the elbow method doesn’t show the elbow point.

The value of the Silhouette score ranges from -1 to 1. Following is the interpretation of the Silhouette score.

- 1: Points are perfectly assigned in a cluster and clusters are easily distinguishable.
- 0: Clusters are overlapping.
- -1: Points are wrongly assigned in a cluster.
```
Silhouette Score = (b-a)/max(a,b)
```
a = average intra-cluster distance, i.e the average distance between each point within a cluster.<br>
b = average inter-cluster distance i.e the average distance between all clusters.
<br>
how to choose best k?
<br>
- For a particular K, all the clusters should have a Silhouette score greater than the average score of the data set represented by the red-dotted line.
- There shouldn’t be wide fluctuations in the size of the clusters. The width of the clusters represents the number of data points.
![alt text](<KMeans Plots/sil_8.png>)
![alt text](<KMeans Plots/silh_19.png>)

k = 8 have cluster 2 double the width of cluster 4 so it's bad choice<br>
k = 19 is better because every cluster silhouette score is above average and clusters have similar width.

## Evaluation
We will evaluate our clustering by the Test Data but how? <br>
Every Point in Test data is assigned to the cluster which it's centroid is the closest centroid to the point, Then we have clusters for Test Data.<br>
**Note:** The centroids is obtained from the k-means training on Training set

We will map each point in each cluster to its ground truth (y_test)
```python
def mapping(k,y_pred,y_actual):
    clusters=[[] for _ in range(k)]
    for i in range(len(y_pred)):
        clusters[y_pred[i]-1].append((int) (y_actual[i]))
    return clusters
```
**Then we will use the following metrics to evaluate our cluserting:**
### 1. Precision
In the context of k-means clustering, precision indicates how **pure** each cluster is, with higher precision values indicating that the clusters contain predominantly similar data points.<br>
Precision(Ci​)=Total number of data points assigned to cluster Ci​Number of data points correctly assigned to cluster Ci​​
### 2. Recall

### 3. F Score

### 4. Conditional Entropy


<!DOCTYPE html>
<html>

<body>

<table >
<tr>
    <td><strong>Metrics<strong></td>
    <td><strong>Precision<strong></td>
    <td><strong>Recall<strong></td>
    <td><strong>F Score<strong></td>
    <td><strong>Condititional Entropy<strong></td>
  </tr>
  <tr>
  <td><strong>DBSCAN<strong></td>
    <td>92.15%</td>
    <td>71.00%</td>
    <td>76.62%</td>
    <td>0.988</td>
  </tr>
  <tr>
    <td><strong>Spectral Clustering<strong></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><strong>K-means<strong></td>
    <td>47.31%</td>
    <td>84.24%</td>
    <td>56.68%</td>
    <td>1.65</td>
  </tr>
</table>

</body>
</html>