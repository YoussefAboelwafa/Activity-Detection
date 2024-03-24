# Activity Detection
Activity Detection using Unsupervised Learning Algorithms: **DBSCAN**, **K-Means** & **Spectral Clustering**

## Dataset
Daily & Sports Activity [Dataset](https://www.kaggle.com/datasets/obirgul/daily-and-sports-activities/data)

#### Brief Description of the Dataset:

Each of the 19 activities is performed by eight subjects (4 female, 4 male, between the ages 20 and 30) for 5 minutes. <br>
Total signal duration is 5 minutes for each activity of each subject.<br>
The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity. <br>

#### The 19 activities are:
- sitting (A1)
- standing (A2)
- lying on back and on right side (A3 and A4)
- ascending and descending stairs (A5 and A6)
- standing in an elevator still (A7)
- and moving around in an elevator (A8)
- walking in a parking lot (A9)
- walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11)
- running on a treadmill with a speed of 8 km/h (A12)
- exercising on a stepper (A13)
- exercising on a cross trainer (A14)
- cycling on an exercise bike in horizontal and vertical positions (A15 and A16)
- rowing (A17)
- jumping (A18)
- and playing basketball (A19)

#### File structure:

- 19 activities (a) (in the order given above)
- 8 subjects (p)
- 60 segments (s)
- 5 units on torso (T), right arm (RA), left arm (LA), right leg (RL), left leg (LL)
- 9 sensors on each unit (x,y,z accelerometers, x,y,z gyroscopes, x,y,z magnetometers)

Folders a01, a02, …, a19 contain data recorded from the 19 activities. <br>

For each activity, the subfolders p1, p2, …, p8 contain data from each of the 8 subjects. <br>

In each subfolder, there are 60 text files s01, s02, …, s60, one for each segment.

## DBSCAN

- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm that is used for clustering tasks. Unlike other clustering algorithms such as K-Means, DBSCAN does not require the user to specify the number of clusters in advance.

- DBSCAN works by defining a neighborhood around each data point. If there are at least a minimum number of points (`min_samples`) within a certain distance (`eps`) of a data point, that data point is considered a core point. Points that are within the `eps` distance of a core point, but do not have `min_samples` within their own `eps` distance, are considered to be border points. All other points are considered noise points.

- DBSCAN has several advantages over other clustering algorithms. It can find arbitrarily shaped clusters, it has a notion of noise, and it does not require the user to specify the number of clusters. However, it can be sensitive to the settings of `eps` and `min_samples`, and it does not perform well when the clusters have different densities.

<hr>

**Pros:**
- DBSCAN can find arbitrarily shaped clusters
- DBSCAN has a notion of noise & is robust to outliers
- DBSCAN does not require the user to specify the number of clusters

**Cons:**
- DBSCAN can be sensitive to the settings of `eps` and `min_samples`
- DBSCAN does not perform well when the clusters have different densities
- DBSCAN does not work well with high-dimensional data
<hr>

#### DBSCAN Algorithm:
This code is an implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm. Here's a step-by-step explanation:

1. **Initialize labels**: The `labels` array is initialized with zeros. This array will hold the cluster assignments for each point in the dataset `X`. A label of `0` means the point has not been assigned to a cluster yet.

2. **Initialize cluster counter**: `C` is a counter for the current cluster. It starts at `0` and is incremented each time a new cluster is created.

3. **Loop over points**: The main loop iterates over each point in the dataset.

4. **Skip labeled points**: If a point has already been assigned to a cluster (i.e., its label is not `0`), it is skipped.

5. **Find neighbors**: For each point, it finds the indices of the points in `X` that are within a distance `eps` from the current point. These points are considered the current point's neighbors.

6. **Mark as noise**: If a point has fewer than `min_samples` neighbors, it is considered noise and is assigned a label of `-1`.

7. **Create new cluster**: If a point has at least `min_samples` neighbors, a new cluster is created. The cluster counter `C` is incremented, and the current point is assigned to the new cluster.

8. **Expand cluster**: The algorithm then attempts to expand the cluster by iterating over the current point's neighbors. If a neighbor is noise, it is reassigned to the current cluster. If a neighbor is unlabeled, it is assigned to the current cluster, and its own neighbors are added to the set of points to be visited. This process continues until no more points can be added to the cluster.

9. **Return labels**: Finally, the function returns the `labels` array, which holds the final cluster assignments for each point in the dataset.

This implementation of DBSCAN treats noise points (outliers) as their own separate cluster, which is a common approach in density-based clustering.

#### Pseudocode:
```python
def get_DBSCAN(X, min_samples, eps)
    Initialize labels[] = 0 for each point in X

    Set C = 0

    for each point i in X
        if labels[i] != 0
            continue to next point

        Find neighbors of i within distance eps

        if number of neighbors < min_samples
            Set labels[i] = -1 (mark as noise)
            continue to next point

        Increment C by 1

        Set labels[i] = C

        Initialize set S with neighbors of i

        Set i = 0
        while i < size of S
            Set j = S[i]
            if labels[j] == -1
                Set labels[j] = C
            else if labels[j] == 0
                Set labels[j] = C
                Find neighbors of j within distance eps
                if number of neighbors >= min_samples
                    Add neighbors of j to S, excluding points already in S
            Increment i by 1

    return labels

```
#### Difference between KMeans and DBSCAN:


<hr>