# acquire.py

from utils.imports import *

def acquire():
    '''
    Acquires zillow data from mySQL using the python module to connect and query. 
    Outputs a single dataframe. 
    Include's the logerror.
    Uses all the tables in the database and all fields related to the properties that are available.
    Only includes properties with a transaction in 2017, and only the last transaction for each property (so no duplicate property ID's), along with zestimate error and date of transaction.
    Only includes properties that include a latitude and longitude value.
    SQL by: Zach Gulde (https://github.com/CodeupClassroom/innis-clustering-exercises)
    '''

    query = '''
    SELECT
        prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        landuse.propertylandusedesc,
        story.storydesc,
        construct.typeconstructiondesc
    FROM properties_2017 prop
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
    LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
    LEFT JOIN storytype story USING (storytypeid)
    LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
    WHERE prop.latitude IS NOT NULL
    AND prop.longitude IS NOT NULL
    AND transactiondate <= '2017-12-31'
    '''
    
    if os.path.exists('/Users/hinzlehome/codeup-data-science/clustering-exercises/csvs/zillow.csv'):
        df = pd.read_csv('/Users/hinzlehome/codeup-data-science/clustering-exercises/csvs/zillow.csv')
    else:
        url=get_db_url('zillow')
        df = pd.read_sql(query, url)
        df.to_csv('/Users/hinzlehome/codeup-data-science/clustering-exercises/csvs/zillow.csv', index=False)
    return df

def overview(df):
    '''
    Code by: Zach Gulde (https://github.com/CodeupClassroom/innis-clustering-exercises)
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    '''
    Code by: Zach Gulde (https://github.com/CodeupClassroom/innis-clustering-exercises)
    '''
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    '''
    Code by: Zach Gulde (https://github.com/CodeupClassroom/innis-clustering-exercises)
    '''
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def acquire_show():
    df=acquire()
    overview(df)
    nulls_by_columns(df)
    nulls_by_rows(df)
    return df