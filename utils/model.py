############################# Modeling ################################   
##################### Show Clusters/Centroids #########################

def cluster(df, feature1, feature2, k):
    X = df[[feature1, feature2]]

    kmeans = KMeans(n_clusters=k).fit(X)
    
    df['cluster'] = kmeans.labels_
    df.cluster = df.cluster.astype('category')
    
    df['cluster'] = kmeans.predict(X)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    df.groupby('cluster')[feature1, feature2].mean()
    
    plt.figure(figsize=(9, 7))
    
    for cluster, subset in df.groupby('cluster'):
        plt.scatter(subset[feature2], subset[feature1], label='cluster ' + str(cluster), alpha=.6)
    
    centroids.plot.scatter(y=feature1, x=feature2, c='black', marker='x', s=100, ax=plt.gca(), label='centroid')
    
    plt.legend()
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Visualizing Cluster Centers')

    return

################ Find The Best K Value For Clustering ##################

def inertia(df, feature1, feature2, r1, r2):
    cols = [feature1, feature2]
    X = df[cols]
    
    inertias = {}
    
    for k in range(r1, r2):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias[k] = kmeans.inertia_
    
    pd.Series(inertias).plot(xlabel='k', ylabel='Inertia', figsize=(9, 7))
    plt.grid()
    return
