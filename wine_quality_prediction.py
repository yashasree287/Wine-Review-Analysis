#import libraries here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
fromlearn.metrics 
import r2_score, mean_squared_error

%matplotlib inline

#read data to DataFrame
wine_main = pd.read_csv('winemag-data-130k-v2.csv')
#Total number of records
n_records = wine_main.shape[0]
n_features = wine_main.shape[1]
# Print the results
print("Total number of records: {}".format(n_records))
print("Total number of features: {}".format(n_features))
wine_main.head()

#plot correlation between features
sns.heatmap(wine_main.corr(), annot=True, fmt='.2f', cmap="coolwarm",square=True)

#identify missing data by column missing_by_columns = 
wine_main.isnull().mean().sort_values(ascending=False).round(4)*100 
missing_by_columns.plot(kind='bar') 
plt.title("% of missing data") 
missing_by_columns

#drop region_2, designation, taster_twitter_handle, taster_name 
wine_clmn_reduced = wine_main.drop(columns=["region_2", "designation", "taster_twitter_handle", "taster_name"]) 
wine_clmn_reduced.head()

#missing by rows
#drop rows with missing price
wine_row_reduced = wine_clmn_reduced.dropna(subset=['price'], how='any')
print(wine_row_reduced.shape) 
wine_row_reduced.head() 

#check missing values by column again
wine_row_reduced.isnull().sum().sort_values(ascending=False)

#Since country is one of the most important attribute to predict price. We will drop 59 rows with missing country. 
wine_row_reduced = wine_row_reduced.dropna(subset=['country','variety'], how='any')
#check missing values by column again
print("Number of rows, columns:", (wine_row_reduced.shape[0], wine_row_reduced.shape[1]))
wine_row_reduced.isnull().sum().sort_values(ascending=False)

wine_row_reduced["region_1"].replace(np.nan, 0, inplace=True) 
wine_row_reduced.isnull().sum()

#Lets group by wines by title
wine_groupped = wine_row_reduced.groupby('title').agg(dict(country ='first', description=list,points = 'mean', price = 'mean', province='first', region_1 = 'first',variety ='first',winery='first')).reset_index() 
wine_groupped

years1 = wine_groupped['title'].str.extract('([1][9][0-9][0-9])').astype('float64') 
years1.fillna(0, inplace=True)
years2 = wine_groupped['title'].str.extract('([2][0][0-1][0-9])').astype('float64') 
years2.fillna(0, inplace=True) 
years = np.add(years1,years2)
wine_groupped = wine_groupped.assign(year = years)
def add_feature_from_title(df, feature):
    '''
    This function search for listed word in title and add it as a separate feauture (column)     
    Args:df (DataFrame): dataframe with column title
    feature (str): word that wull be searched in title         
    Return:df     '''
    
    feature_occurrence = df['title'].str.lower().str.find(feature)
    feature_occurrence[feature_occurrence >= 0] = 1
    feature_occurrence[feature_occurrence == -1] = 0
    df[feature] = feature_occurrence
    return df
#add feature sparkling and vintage
wine_groupped = add_feature_from_title(wine_groupped, "sparkling")
wine_groupped = add_feature_from_title(wine_groupped, "vintage")
#check if there is vintage wine with no year assigned
wine_groupped[wine_groupped["year"]==0][wine_groupped['vintage']==1]

#Calculate mean of year for sparkling vintage wines mean_vintage_sparkling = 
round(wine_groupped[wine_groupped["vintage"]==1] [wine_groupped['sparkling']==1]['year'].mean(),0)
mean_vintage_sparkling

#substitute missing year
index = wine_groupped[wine_groupped["year"]==0]
[wine_groupped['vintage']==1].index[0]
wine_groupped.set_value(index=index, col='year', value=mean_vintage_sparkling)

#now assign the latest year for all 0 year values
wine_groupped['year'].replace(0, 2017, inplace=True)
wine_groupped[:5]
 
#identify categorical attributes 
wine_groupped.dtypes

print("Number of countries: ", len(wine_groupped['country'].value_counts()))
print("Number of provinces: ", len(wine_groupped['province'].value_counts()))
print("Number of regions: ", len(wine_groupped['region_1'].value_counts()))
print("Number of varieties: ", len(wine_groupped['variety'].value_counts()))
print("Number of wineries: ", len(wine_groupped['winery'].value_counts()))
df_no_reviews = wine_groupped.drop(['title', 'description', 'winery', 'region_1'], axis=1)

#check later, maybe drop regions (1204),
#decided to drop regions

# Re-encode categorical features to be kept in the analysis.
reencode_categ = ['country', 'province', 'variety']
reencoded_wines = pd.get_dummies(df_no_reviews, 
columns=reencode_categ, drop_first=True)
print(df_no_reviews.shape)
print(reencoded_wines.shape)
def clean_data(df):
    '''
    Perform feature trimming, re-encoding, and engineering for winery data     
    Input
    df (DataFrame): winery data frame
    Return (DataFrame): Cleaned winery data frame
    '''
    #drop columns
    df_dropped = df.drop(columns=["region_1", "region_2", "designation", "taster_twitter_handle","taster_name", "winery", "description"])
    #drop rows
    df_dropped = df_dropped.dropna(subset=['price','country','variety'], how='any')     
    #group data by title
    df_groupped = df_dropped.groupby('title').agg(dict(country = 'first', points = 'mean', price = 'mean', province='first', variety = 'first')).reset_index()
    
    #extract year
    years1 = df_groupped['title'].str.extract('([1][9][0-9][0-9])').astype('float64')
    years1.fillna(0, inplace=True)
    years2 = df_groupped['title'].str.extract('([2][0][0-1][0-9])').astype('float64')
    years2.fillna(0, inplace=True)
    years = np.add(years1,years2)
    df_groupped = df_groupped.assign(year = years)
    
    #add feature sparkling and vintage
    df_groupped = add_feature_from_title(df_groupped, "sparkling")
    df_groupped = add_feature_from_title(df_groupped, "vintage")
    
    #Calculate mean of year for sparkling vintage wines
    mean = round(df_groupped[df_groupped["vintage"]==1] [df_groupped['sparkling']==1]['year'].mean(),0)
    
    #substitute missing year for vintage wine
    index = wine_groupped[df_groupped["year"]==0][df_groupped['vintage']==1].index[0]
    df_groupped.set_value(index=index, col='year', value=mean)     
    
    #Usually when there is no year indicated in wine, it means that it was produced in current year
    #replace 0 year with the latest year (2017)
    df_groupped['year'].replace(0, 2017, inplace=True)
    
    #drop title
    df_groupped = df_groupped.drop(['title'], axis=1)
    
    # Re-encode categorical features to be kept in the analysis.
    reencode_categ = ['country', 'province', 'variety']
    df_groupped = pd.get_dummies(df_groupped, columns=reencode_categ, drop_first=True)
    
    return df_groupped
wine_cleaned = clean_data(wine_main)

#check that shape is similar to the shape obtained without cleaning function
print(wine_cleaned.shape)

#check whether all columns are of numeric dtype
for column in wine_cleaned.columns:
    if pd.api.types.is_numeric_dtype(wine_cleaned[column])==False:
        print(column)
        print(wine_cleaned[column])
                                
#check missing values
missing = wine_cleaned.isnull().sum(axis=0).sort_values(ascending=False)
print(missing[:5])

quality = wine_cleaned["points"]
features = wine_cleaned.drop(["points"], axis=1)

# Initialize a scaler, then apply it to the features
scaler = StandardScaler()  # default=(0, 1)
numerical = ['price', 'year']
features[:] = scaler.fit_transform(features[:])

# Show an example of a record with scaling applied 
display(features.head(n = 5))

# Split the 'features' and 'income' data into training and testing 
sets
X_train, X_test, y_train, y_test = train_test_split(features,quality,test_size = 0.2,random_state = 42)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#Initialize the three models
linear_model = LinearRegression()
forest_model = RandomForestRegressor(random_state=42)
ada_model = AdaBoostRegressor(random_state=42)
gboost_model = GradientBoostingRegressor(random_state=42)
def evaluate_model(model):
    '''Evaluate model based on r2 score, mean squared error of
training and test sets
    
    Input: model
    Return: dictionary with scores
    
    '''
    model = model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)
    results = {}
    results['train_r2_score'] = r2_score(y_train, predict_train)
    results['test_r2_score'] = r2_score(y_test, predict_test)
    results['train_mse'] = mean_squared_error(y_train, predict_train)
    results['test_mse'] = mean_squared_error(y_test, predict_test)
    return results
for model in [gboost_model, linear_model, forest_model, ada_model]:
    results = evaluate_model(model)
    print(model)
    print(results)


# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#Initialize the classifier
gboost_model = GradientBoostingRegressor(random_state=42)

#Create the parameters list you wish to tune, using a dictionary if needed.
#parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'loss': ['ls'], 'learning_rate': [0.1, 0.2],'n_estimators': [100], 'max_depth': [ 5, 7], 'min_samples_split': [4, 6]}

#Make an r2_score scoring object using make_scorer()
scorer = make_scorer(r2_score)

# Perform grid search on the regressor using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(gboost_model, parameters, scoring = scorer)

#Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_model = grid_fit.best_estimator_
best_predictions = best_model.predict(X_test)
print(best_model)
print("\nOptimized Model\n------")
print("Final r2 score on the testing data: {:.4f}".format(r2_score(y_test, best_predictions)))
print("Final MSE on the testing data: {:.4f}".format(mean_squared_error(y_test, best_predictions)))
plt.plot(best_predictions, y_test, 'bo');
plt.xlabel('Predicted');
plt.ylabel('Actual')
plt.title('Predicted vs Actual quality, points')
Text(0.5,1,'Predicted vs Actual quality, points')

#Extract the feature importances using .feature_importances_ 
importances =best_model.feature_importances_
print(importances)
pd.Series(importances, index=X_train.columns).sort_values()[-40:].plot(kind='barh', figsize=(5,10));
coefs_df =pd.DataFrame()
coefs_df['est_int'] =X_train.columns
coefs_df['coefs'] = importances
coefs_df['abs_coefs'] = np.abs(importances)
coefs_df.sort_values('abs_coefs', ascending=False).head(20)

# Import functionality for cloning a model
from sklearn.base import clone
 
# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:40]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:40]]]
 
# Train on the "best" model found from grid search earlier
model = (clone(best_model)).fit(X_train_reduced, y_train)
# Make new predictions
reduced_predictions = model.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("r2 score on testing data: {:.4f}".format(r2_score(y_test, best_predictions)))
print("MSE on testing data: {:.4f}".format(mean_squared_error(y_test, best_predictions)))
print("\nFinal Model trained on reduced data\n------")
print("r2 score on testing data: {:.4f}".format(r2_score(y_test, reduced_predictions)))
print("MSE on testing data: {:.4f}".format(mean_squared_error(y_test, reduced_predictions)))






