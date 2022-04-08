import sys
sys.path.insert(0, '/Users/hinzlehome/codeup-data-science/clustering-project/utils')
from imports import *

############################# Modeling ################################   
##################### Make Clusters #########################

def cluster(df,feature1, feature2, k):
    X = df[[feature1, feature2]]

    kmeans = KMeans(n_clusters=k).fit(X)
    
    df['cluster'] = kmeans.labels_
    df.cluster = df.cluster.astype('category')
    
    df['cluster'] = kmeans.predict(X)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    print(centroids)
    
    df.groupby('cluster')[feature1, feature2].mean()
    
    plt.figure(figsize=(9, 7))
    
    for cluster, subset in df.groupby('cluster'):
        plt.scatter(subset[feature2], subset[feature1], label='cluster ' + str(cluster), alpha=.6)
    
    centroids.plot.scatter(y=feature1, x=feature2, c='black', marker='x', s=100, ax=plt.gca(), label='centroid')
    
    plt.legend()
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Visualizing Cluster Centers')

    return df, centroids

################ Find The Best K Value For Clustering ##################

def inertia(df, cols, r1, r2):

    X = df[cols]
    
    inertias = {}
    
    for k in range(r1, r2):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias[k] = kmeans.inertia_
    
    pd.Series(inertias).plot(xlabel='k', ylabel='Inertia', figsize=(9, 7))
    plt.grid()
    return

############################# Regression  ################################

def reg_zillow_train(X_train,y_train,X_validate,y_validate):

	y_train['yhat_base_prop']=y_train.logerror.mean()


	# evaluate: rmse
	mse_baseline = mean_squared_error(y_train.logerror, y_train.yhat_base_prop)

	model = LinearRegression().fit(X_train[['cluster']],y_train['logerror'])

	y_train['yhat_prop_ols'] = model.predict(X_train[['cluster']])

	mse_ols = mean_squared_error(y_train.logerror, y_train.yhat_prop_ols)

	print("MSE OLS sklearn: ","{:.2e}".format(mse_ols),"\n") 

	if mse_ols-mse_baseline<0:
		print("y_hat_ols superior","\n")
	else:
		print("yhat_baseline superior","\n")

	lars = LassoLars(alpha=1.0)

	# create the model object

	lars.fit(X_train[['cluster']], y_train.logerror)

	# fit the model to our training data. We must specify the column in y_train, 
	# since we have converted it to a dataframe from a series!

	y_train['yhat_prop_lars'] = lars.predict(X_train[['cluster']])

	# predict train

	# evaluate: rmse
	mse_lars_train = mean_squared_error(y_train.logerror, y_train.yhat_prop_lars)


	print("MSE for Lasso + Lars\nTraining/In-Sample: ","{:.2e}".format(mse_lars_train),"\n")

	if mse_lars_train-mse_baseline<0:
		print("y_hat_lars superior","\n")
	else:
		print("yhat_baseline superior","\n")


	# create the model object
	glm = TweedieRegressor(power=1, alpha=0)

	# fit the model to our training data. We must specify the column in y_train, 
	# since we have converted it to a dataframe from a series! 
	glm.fit(X_train[['cluster']], y_train.logerror)

	# predict train
	y_train['yhat_prop_glm'] = glm.predict(X_train[['cluster']])

	# evaluate: mse
	mse_glm_train = mean_squared_error(y_train.logerror, y_train.yhat_prop_glm)

	print("MSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ","{:.2e}".format( mse_glm_train),"\n")

	if mse_glm_train-mse_baseline<0:
		print("y_hat superior","\n")
	else:
		print("yhat_baseline superior","\n")

	# make the polynomial features to get a new set of features
	pf = PolynomialFeatures(degree=2)

	# fit and transform X_train_scaled
	X_train_degree2 = pf.fit_transform(X_train[['cluster']])

	# create the model object
	lmp = LinearRegression(normalize=True)

	# fit the model to our training data. We must specify the column in y_train, 
	# since we have converted it to a dataframe from a series! 
	lmp.fit(X_train_degree2,y_train.logerror)

	# predict train
	y_train['yhat_prop_lmp'] = lmp.predict(X_train_degree2)

	# evaluate: rmse
	mse_lmp_train = mean_squared_error(y_train.logerror, y_train.yhat_prop_lmp)

	print("MSE for Polynomial Model, degrees=2\nTraining/In-Sample: ","{:.2e}".format( mse_lmp_train),"\n")

	if mse_glm_train-mse_baseline<0:
		print("y_hat superior","\n")
	else:
		print("yhat_baseline superior","\n")
		
	mses=pd.DataFrame([mse_baseline,
	mse_ols,
	mse_lars_train,
	mse_glm_train,
	mse_lmp_train],
	columns=['mse'],
	index= ['mse_baseline',
	'mse_ols',
	'mse_lars_train',
	'mse_glm_train',
	'mse_lmp_train'])

	X_validate,centroids=cluster(X_validate,'calculatedfinishedsquarefeet','taxvaluedollarcnt',2)

	# transform X_validate_scaled 
	X_validate_degree2 = pf.transform(X_validate[['cluster']])

	y_validate['yhat_val_base'] = y_validate.logerror.mean()

	y_validate['yhat_val_ols'] = model.predict(X_validate[['cluster']])

	y_validate['yhat_val_lars'] = lars.predict(X_validate[['cluster']])

	y_validate['yhat_val_glm'] = glm.predict(X_validate[['cluster']])

	y_validate['yhat_val_lmp'] = lmp.predict(X_validate_degree2)



	mse_base_val = mean_squared_error(y_validate.logerror, y_validate.yhat_val_base)

	mse_ols_val = mean_squared_error(y_validate.logerror, y_validate.yhat_val_ols)

	mse_lars_val = mean_squared_error(y_validate.logerror, y_validate.yhat_val_lars)

	mse_glm_val = mean_squared_error(y_validate.logerror, y_validate.yhat_val_glm)

	mse_lmp_val = mean_squared_error(y_validate.logerror, y_validate.yhat_val_lmp)

	mse_val=[mse_base_val,mse_ols_val,mse_lars_val,mse_glm_val,mse_lmp_val]
	mse_val=pd.DataFrame(mse_val,index=['mse_base_val','mse_ols_val','mse_lars_val','mse_glm_val','mse_lmp_val' ],columns=['mse'])

	print(mses,"\n")
	print(mse_val,"\n")
	print(y_train.mean(),"\n")
	print(y_validate.mean(),"\n")

	return pf,model,