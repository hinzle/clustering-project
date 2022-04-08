import sys
sys.path.insert(0, '/Users/hinzlehome/codeup-data-science/clustering-project/utils')
from imports import *



def wrangle_zillow(use_cache=True):
    '''
    Acquires and prepares zillow 2017 property data for exploration and modeling.
    '''
    if os.path.exists('/Users/hinzlehome/codeup-data-science/regression-project/csvs/zillow_db.csv') and use_cache:
        print('Using cached csv')
        df=pd.read_csv('/Users/hinzlehome/codeup-data-science/regression-project/csvs/zillow_db.csv')
    
    else:
        print('Acquiring data from SQL database')
        df=pd.read_sql(
            '''
            SELECT *
            FROM properties_2017
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            ''',
                # LEFT JOIN contract_types USING (contract_type_id)
                # LEFT JOIN payment_types USING (payment_type_id)
                # ''',
            get_db_url('zillow'))
        df.to_csv('/Users/hinzlehome/codeup-data-science/regression-project/csvs/zillow_db.csv', index=False)
    cols=['bedroomcnt','bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips', 'propertylandusedesc']
    df=df[cols]
    df=df[df.propertylandusedesc=='Single Family Residential']
    df=df.replace(r'^\s*$', np.nan, regex=True)
    df=df.dropna()
    df=df.drop('propertylandusedesc',axis=1)
    df=df.drop_duplicates()
    
    df=df.rename(
        {
        'yearbuilt':'year',
        'bedroomcnt':'beds',
        'bathroomcnt':'baths',
        'calculatedfinishedsquarefeet':'sqft',
        'taxvaluedollarcnt':'property_value',
        'taxamount':'taxes'
        },axis=1)
    # df.beds=df.beds.astype('Int64')
    # df.baths=df.baths.astype('Int64')
    df.year=df.year.astype('category')
    df.fips=df.fips.astype('category')
    df.to_csv('/Users/hinzlehome/codeup-data-science/regression-project/csvs/pre_split_zillow.csv', index=False)

    '''
    takes in a dataframe and a target name, outputs three dataframes: 'train', 'validate', 'test', each stratified on the named target. 

    ->: str e.g. 'df.target_column'
    <-: 3 x pandas.DataFrame ; 'train', 'validate', 'test'

    training set is 60% of total sample
    validate set is 23% of total sample
    test set is 17% of total sample

    '''
    train, _ = train_test_split(df, train_size=.6, random_state=123)
    validate, test = train_test_split(_, test_size=(3/7), random_state=123)
    train=train.reset_index(drop=True)
    validate=validate.reset_index(drop=True)
    test=test.reset_index(drop=True)

    '''
    Scales the 3 data splits.

    takes in the train, validate, and test data splits and returns their scaled counterparts.

    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['beds', 'baths', 'taxes', 'sqft']

    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    # dont scale the target

    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    

    return [train_scaled, validate_scaled, test_scaled]


def zillow_data():
    '''
    This function uses a SQL query to access the Codeup MySQL database and join 
    together all the relevant data from the zillow database.
    The data obtained includes all properties in the dataset which had a transaction in 2017.
    The function caches a csv in the local directory for later use. 
    '''
    # establish a filename for the local csv
    filename = 'zillow.csv'
    # check to see if a local copy already exists. 
    if os.path.exists(filename):
        print('Reading from local CSV...')
        # if so, return the local csv
        return pd.read_csv(filename)
    # otherwise, pull the data from the database:
    # establish database url
    url = env.get_db_url('zillow')
    # establish query
    sql = '''
            SELECT prop.*,
                   pred.logerror,
                   const.typeconstructiondesc,
                   arch.architecturalstyledesc,
                   land.propertylandusedesc,
                   heat.heatingorsystemdesc,
                   air.airconditioningdesc, 
                   bldg.buildingclassdesc,
                   story.storydesc
              FROM properties_2017 prop
                JOIN predictions_2017            pred  USING(parcelid)
                LEFT JOIN typeconstructiontype   const USING(typeconstructiontypeid)
                LEFT JOIN architecturalstyletype arch  USING(architecturalstyletypeid)
                LEFT JOIN propertylandusetype    land  USING(propertylandusetypeid)
                LEFT JOIN heatingorsystemtype    heat  USING(heatingorsystemtypeid)
                LEFT JOIN airconditioningtype    air   USING(airconditioningtypeid)
                LEFT JOIN buildingclasstype      bldg  USING(buildingclasstypeid)
                LEFT JOIN storytype              story USING(storytypeid)
              WHERE pred.transactiondate LIKE "2017%%"
                AND pred.transactiondate in (
                                             SELECT MAX(transactiondate)
                                               FROM predictions_2017
                                               GROUP BY parcelid
                                             )
                AND prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL;
            '''
    print('No local file exists\nReading from SQL database...')
    # query the database and return the resulting table as a pandas dataframe
    df = pd.read_sql(sql, url)
    # save the dataframe to the local directory as a csv
    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    # return the resulting dataframe
    return df



def prep_zillow(df):
    # drop redundant id code columns
    id_cols = [col for col in df.columns if 'typeid' in col or col in ['id', 'parcelid']]
    df = df.drop(columns=id_cols)
    # filter for single family properties
    df = df[df.propertylandusedesc == 'Single Family Residential']
    # drop specified columns
    cols_to_drop = ['calculatedbathnbr',
                    'finishedfloor1squarefeet',
                    'finishedsquarefeet12', 
                    'regionidcity',
                    'landtaxvaluedollarcnt',
                    'taxamount',
                    'rawcensustractandblock',
                    'roomcnt',
                    'regionidcounty']
    df = df.drop(columns=cols_to_drop)
    # fill null values with 0 in specified columns
    cols_to_fill_zero = ['fireplacecnt',
                         'garagecarcnt',
                         'garagetotalsqft',
                         'hashottuborspa',
                         'poolcnt',
                         'threequarterbathnbr',
                         'taxdelinquencyflag']
    for col in cols_to_fill_zero:
        df[col] = np.where(df[col].isna(), 0, df[col]) 
    # drop columns with more than 5% null values
    for col in df.columns:
        if df[col].isnull().mean() > .05:
            df = df.drop(columns=col)
    # drop rows that remain with null values
    df = df.dropna()   
    # changing numeric codes to strings
    df['fips'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['regionidzip'] = df.regionidzip.apply(lambda x: str(int(x)))
    df['censustractandblock'] = df.censustractandblock.apply(lambda x: str(int(x)))
    # change the 'Y' in taxdelinquencyflag to 1
    df['taxdelinquencyflag'] = np.where(df.taxdelinquencyflag == 'Y', 1, df.taxdelinquencyflag)
    # change boolean column to int
    df['hashottuborspa'] = df.hashottuborspa.apply(lambda x: str(int(x)))
    # changing year from float to int
    df['yearbuilt'] = df.yearbuilt.apply(lambda x: int(x))
    df['assessmentyear'] = df.yearbuilt.apply(lambda x: int(x))
    # moving the latitude and longitude decimal place
    df['latitude'] = df.latitude / 1_000_000
    df['longitude'] = df.longitude / 1_000_000
    # adding a feature: age 
    df['age'] = 2017 - df.yearbuilt
    # add a feature: has_garage
    df['bool_has_garage'] = np.where(df.garagecarcnt > 0, 1, 0)
    # add a feature: has_pool
    df['bool_has_pool'] = np.where(df.poolcnt > 0, 1, 0)
    # add a feature: has_fireplace
    df['bool_has_fireplace'] = np.where(df.fireplacecnt > 0, 1, 0)
    # add a feature: taxvalue_per_sqft
    df['taxvalue_per_sqft'] = df.taxvaluedollarcnt / df.calculatedfinishedsquarefeet
    # add a feature: taxvalue_per_bedroom
    df['taxvalue_per_bedroom'] = df.taxvaluedollarcnt / df.bedroomcnt
    #add a feature: taxvalue_per_bathroom
    df['taxvalue_per_bathroom'] = df.taxvaluedollarcnt / df.bathroomcnt    
    #add a feature: taxvalue_per_room
    df['taxvalue_per_bathroom'] = df.taxvaluedollarcnt / (df.bathroomcnt + df.bedroomcnt)
    # adding prefix to boolean columns
    df = df.rename(columns={'hashottuborspa': 'bool_hashottuborspa'})
    df = df.rename(columns={'taxdelinquencyflag': 'bool_taxdelinquencyflag'})
    # rename sqft column
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sqft'})
    # add a column: absolute value of logerror (derived form target)
    df['abs_logerror'] = abs(df.logerror)
    # add a column: direction of logerror (high or low) (derived from target)
    df['logerror_direction'] = np.where(df.logerror < 0, 'low', 'high')


    return df

def train_validate_test_split(df, test_size=.2, validate_size=.3, random_state=42):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    # split the dataframe into train and test
    train, test = train_test_split(df, test_size=.2, random_state=42)
    # further split the train dataframe into train and validate
    train, validate = train_test_split(train, test_size=.3, random_state=42)
    # print the sample size of each resulting dataframe
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test

def remove_outliers(train, validate, test, k, col_list):
    ''' 
    This function takes in a dataset split into three sample dataframes: train, validate and test.
    It calculates an outlier range based on a given value for k, using the interquartile range 
    from the train sample. It then applies that outlier range to each of the three samples, removing
    outliers from a given list of feature columns. The train, validate, and test dataframes 
    are returned, in that order. 
    '''
    # Create a column that will label our rows as containing an outlier value or not
    train['outlier'] = False
    validate['outlier'] = False
    test['outlier'] = False
    for col in col_list:

        q1, q3 = train[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        train['outlier'] = np.where(((train[col] < lower_bound) | (train[col] > upper_bound)) & (train.outlier == False), True, train.outlier)
        validate['outlier'] = np.where(((validate[col] < lower_bound) | (validate[col] > upper_bound)) & (validate.outlier == False), True, validate.outlier)
        test['outlier'] = np.where(((test[col] < lower_bound) | (test[col] > upper_bound)) & (test.outlier == False), True, test.outlier)

    # remove observations with the outlier label in each of the three samples
    train = train[train.outlier == False]
    train = train.drop(columns=['outlier'])

    validate = validate[validate.outlier == False]
    validate = validate.drop(columns=['outlier'])

    test = test[test.outlier == False]
    test = test.drop(columns=['outlier'])

    # print the remaining 
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test

def scale_zillow(train, validate, test, target, scaler_type=MinMaxScaler()):
    '''
    This takes in the train, validate, and test dataframes, as well as the target label. 
    It then fits a scaler object to the train sample based on the given sample_type, applies that
    scaler to the train, validate, and test samples, and appends the new scaled data to the 
    dataframes as additional columns with the prefix 'scaled_'. 
    train, validate, and test dataframes are returned, in that order. 
    '''
    # identify quantitative features to scale
    quant_features = [col for col in train.columns if (train[col].dtype != 'object') 
                                                    & (target not in col) 
                                                    & ('bool_' not in col)]
    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)
    # screate and fit the scaler
    scaler = scaler_type.fit(train[quant_features])
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
    # add 'scaled' prefix to columns
    for feature in quant_features:
        train_scaled = train_scaled.rename(columns={feature: f'scaled_{feature}'})
        validate_scaled = validate_scaled.rename(columns={feature: f'scaled_{feature}'})
        test_scaled = test_scaled.rename(columns={feature: f'scaled_{feature}'})
    # concat scaled feature columns to original train, validate, test df's
    train = pd.concat([train, train_scaled], axis=1)
    validate = pd.concat([validate, validate_scaled], axis=1)
    test = pd.concat([test, test_scaled], axis=1)

    return train, validate, test

def encode_zillow(train, validate, test, target):
    '''
    This function takes in the train, validate, and test samples, as well as a label for the target variable. 
    It then encodes each of the categorical variables using one-hot encoding with dummy variables and appends 
    the new encoded variables to the original dataframes as new columns with the prefix 'enc_{variable_name}'.
    train, validate and test dataframes are returned (in that order)
    '''
    # identify the features to encode (categorical features represented by non-numeric data types)
    features_to_encode = [col for col in train.columns if (train[col].dtype == 'object')
                                                        & ('bool_' not in col) 
                                                        & (target not in col)
                                                        & (train[col].nunique() < 25)]
    #iterate through the list of features                  
    for feature in features_to_encode:
        # establish dummy variables
        dummy_df = pd.get_dummies(train[feature],
                                  prefix=f'enc_{train[feature].name}',
                                  drop_first=True)
        # add the dummies as new columns to the original dataframe
        train = pd.concat([train, dummy_df], axis=1)

    # then repeat the process for the other two samples:

    for feature in features_to_encode:
        dummy_df = pd.get_dummies(validate[feature],
                                  prefix=f'enc_{validate[feature].name}',
                                  drop_first=True)
        validate = pd.concat([validate, dummy_df], axis=1)
        
    for feature in features_to_encode:
        dummy_df = pd.get_dummies(test[feature],
                                  prefix=f'enc_{test[feature].name}',
                                  drop_first=True)
        test = pd.concat([test, dummy_df], axis=1)
    
    return train, validate, test

def add_clusters(train, validate, test):
    
    # cluster_BedBath

    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt']
    x = train[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBath'] = kmeans.predict(x)
        sample['cluster_BedBath'] = sample.cluster_BedBath.map({1:'low', 0:'mid', 2:'high'})


    # cluster_BedBathSqft

    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt', 'scaled_sqft']
    x = train[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBathSqft'] = kmeans.predict(x)
        sample['cluster_BedBathSqft'] = sample.cluster_BedBathSqft.map({1:'low', 0:'mid', 2:'high'})

    # cluster_LatLong
    features = ['scaled_latitude', 'scaled_longitude']
    x = train[features]
    kmeans = KMeans(n_clusters=4, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_LatLong'] = kmeans.predict(x)
        sample['cluster_LatLong'] = sample.cluster_LatLong.map({0:'east', 1:'central', 2:'west', 3:'north'})

    # cluster_BedBathTaxvaluepersqft
    features = ['scaled_bedroomcnt', 'scaled_bathroomcnt', 'scaled_taxvalue_per_sqft']
    x = train[features]
    kmeans = KMeans(n_clusters=3, random_state=random_state)
    kmeans.fit(x)

    for sample in [train, validate, test]:
        x = sample[features]
        sample['cluster_BedBathTaxvaluepersqft'] = kmeans.predict(x)
        sample['cluster_BedBathTaxvaluepersqft'] = sample.cluster_BedBathTaxvaluepersqft.astype(str)

    return train, validate, test


def ml_data(train, validate, test, target=list):
	'''
	->: train, validate, test 
	<-: X_train, y_train, X_validate, y_validate, X_test, y_test
	'''
	X_train = train.drop(columns=target)
	y_train = train[target]
	X_validate = validate.drop(columns=target)
	y_validate = validate[target]
	X_test = test.drop(columns=target)
	y_test = test[target]
	return [X_train, y_train, X_validate, y_validate, X_test, y_test]

def residuals(actual, predicted):
    '''
    ∆(y,yhat)
    '''
    return actual - predicted

def sse(actual, predicted):
    '''
    sum of squared error
    '''
    return (residuals(actual, predicted) ** 2).sum()

def mse(actual, predicted):
    '''
    mean squared error
    '''
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    '''
    root mean squared error
    '''
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    '''
    explained sum of squared error
    '''
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    '''
    total sum of squared error
    '''
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    '''
    explained variance
    '''
    return ess(actual, predicted) / tss(actual)

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
        'r2': r2_score(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    sse_baseline = sse(actual, actual.mean())
    sse_model = sse(actual, predicted)
    return sse_model < sse_baseline

def evaluate_hypothesis(p: float, alpha: float = 0.05, output: bool = True) -> bool:
    '''
    Compare the p value to the established alpha value to determine if the null hypothesis
    should be rejected or not.
    '''

    if p < alpha:
        if output:
            print('\nReject H_0')
        return False
    else: 
        if output:
            print('\nFail to Reject H_0')
        return True

def pearsons_r_p(data_for_category1, data_for_category2, alpha = 0.05):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship and outputs 
    the relevant information to the console. 
    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas.

    "  
    '''

    # conduct test using scipy.stats.peasonr() test
    r, p = stats.pearsonr(data_for_category1, data_for_category2)

    # output
    print(f'r = {r:.4f}')
    print(f'p = {p:.4f}')

    # evaluate the hypothesis against the established alpha value
    evaluate_hypothesis(p, alpha)

    if r<=0.2:
        print("no correlation")
    elif r<=0.5:
        print("weak correlation")
    elif r<=0.75:
        print("moderate correlation")
    else:
        print("strong correlation")

    # estimate strength of relation

    return r,p