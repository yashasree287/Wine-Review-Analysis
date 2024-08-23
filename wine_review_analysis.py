#import libraries here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import re
from PIL import Image
from sklearn import preprocessing
%matplotlib inline

#read data to DataFrame
wine_main = pd.read_csv('winemag-data-130k-v2.csv')

#shape
print("Number of reviews:", wine_main.shape[0])
print("Number of columns:", wine_main.shape[1])
print("total number of countries: ", 
len(wine_main["country"].value_counts()))
wine_main.head

#identify missing data by column
missing_by_columns = wine_main.isnull().mean().sort_values(ascending=False).round(4)*100
missing_by_columns.plot(kind='bar')
plt.title("% of missing data")
missing_by_columns

#missing by rows
#drop rows with missing price
#Since country is one of the most important attribute to predict price. We will drop 59 rows with missing country. 
wine_row_reduced = wine_clmn_reduced.dropna(subset=['country','variety', 'price'], how='any')

#check missing values by column again
print("Number of rows, columns:", (wine_row_reduced.shape[0], wine_row_reduced.shape[1]))
wine_row_reduced.isnull().sum().sort_values(ascending=False)

#Lets group by wines by title
wine_groupped = wine_row_reduced.groupby('title').agg(dict(country = 'first', description = list, points = 'mean', price = 'mean', province='first', region_1 = 'first', variety = 'first', winery = 'first')).reset_index() 
wine_groupped

countries_wine_counts = wine_groupped['country'].value_counts(ascending=False)
print("total number of countries: ", len(countries_wine_counts))
countries_wine_counts
countries_wine_counts[:12].plot(kind='bar')
plt.title("Major wine producers")
plt.ylabel("Number of wines")
plt.xlabel("Country")
major_countries = countries_wine_counts[:12].index
print("Major wine producers:", major_countries)
wine_groupped['counts'] = 0
country_variety_counts =wine_groupped.groupby(['country','variety']).agg(dict(counts='count',points='media')).sort_values(by='counts',ascending=False).reset_index()
us = country_variety_counts.loc[country_variety_counts['country']=='US'][:10]
variety_list = us['variety'].values
variety_list
array(['Pinot Noir', 'Cabernet Sauvignon', 'Chardonnay', 'Syrah','Red Blend', 'Zinfandel', 'Merlot', 'Sauvignon Blanc','Bordeaux-style Red Blend', 'Riesling'], dtype=object)
wine_groupped.drop(["counts"], axis=1, inplace=True)
wine_groupped[wine_groupped['variety'].isin(variety_list)][wine_groupped['country']=='US']

#create dataframe only with US wines
df = wine_groupped[wine_groupped['variety'].isin(variety_list)][wine_groupped['country']=='US']

#create boxplot with median number of points indicated
fig, ax = plt.subplots(figsize = (15,7))
chart = sns.boxplot(x='variety',y='points', data=df, ax = ax, order=variety_list)
medians = us['points'].values
median_labels = [str(int(s)) for s in medians]
pos = range(len(medians))
for tick,label in zip(pos,ax.get_xticklabels()):
   ax.text(pos[tick], medians[tick] + 1.0, median_labels[tick], horizontalalignment='center', size='medium', color='w', weight='semibold')
plt.xticks(rotation = 90)
plt.title("Quality rating of major US varieties (left to right from most popular to less popular variety) ")
plt.show()

#Create function that will determine 10 top wine varieties for a given country 
#and plot boxplot diagram of median points for these 10 varieties
def top_ten_varieties(df, country):  
    ''' Returns top ten most popular varieties by country'''

    country_data = df.loc[country_variety_counts['country']==country][:10]
    return country_data['variety'].values
def create_points_boxplot(df, country, n_varieties=10):

    '''Creates points boxplot 

    Input
    df: data frame
    country (str): country for which boxplot will be created
    n_varieties: number of varieties to plot


    Return: none'''
    country_data = country_variety_counts.loc[country_variety_counts['country']==country][:n_varieties]
    variety_list = country_data['variety'].values
    df = df[df['variety'].isin(variety_list)][df['country']==country]
    fig, ax = plt.subplots(figsize = (6,5))
    chart = sns.boxplot(x='variety',y='points', data=df, ax = ax, order=variety_list)
    medians = country_data['points'].values
    median_labels = [str(int(s)) for s in medians]
    pos = range(len(medians))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick], medians[tick] + 0.3, median_labels[tick], horizontalalignment='center', size='medium', color='black', weight='semibold')
    plt.xticks(rotation = 90)
    plt.title(country)
    plt.show()  

#Creates boxplot for 12 major countries with 12 most popular varieties 
#x axis from left to right from most popular to least popular
import warnings
warnings.filterwarnings('ignore')
for country in major_countries:
  create_points_boxplot(wine_groupped, country, 15)

#create a scatter plot points vs price
wine_groupped.plot(kind='scatter' , x = 'price',y='points')
plt.title("Points vs Price")
Text(0.5,1,'Points vs Price')

#similar but only for countries from major country list with hue=country
df =wine_groupped[wine_groupped['country'].isin(major_countries)]
sns.lmplot(x = 'price', y='points', hue_order=major_countries, fit_reg=False, palette =sns.color_palette("coloblind"),data=df, hue="country", )
plt.title("Points vs Price for 12 major countries")
Text(0.5,1,'Points vs Price for 12 major countries')

#create pairplot for continues features
sns.set(style="ticks")
sns.pairplot(df, hue_order =major_countries, hue="country" )

#create data frame only for the wines <$15
df1 =wine_groupped[wine_groupped['price']<15].sort_values(['points', 'price'], ascending =[False, True])
df1[['title','variety','country','province','points','price']][:15]

#scatter plot for wines <$15
sns.lmplot(x = 'price', y= 'points', fit_reg=False, data =df1)

#create data frame only for the wines $15<=price<$50
df2 =wine_groupped[wine_groupped['price']>=15][wine_groupped['price']<50.sort_values(['points', 'price'],ascending=[False,True])
df2[['title','variety','country', 'province','points','price' ] ][:15]

#scatter plot wines $15<=price<$50
sns.lmplot(x = 'price', y= 'points', fit_reg=False, data = df2)

#create data frame only for the wines $50<=price<$150
df3 =wine_groupped[wine_groupped['price']>=50][wine_groupped['price']<150].sort_values(['points', 'price'],ascending=[False,True])
df3[['title','variety','country', 'province','points','price' ], ] ][:15]

#scatter plot wines $50<=price<$150
sns.lmplot(x = 'price', y= 'points', fit_reg=False,data = df3)
#create data frame only for the wines with price >$150
df4 =wine_groupped[wine_groupped['price']>=150].sort_values(['points', 'price'], ascending=[False, True])
df4[['title','variety','country','province','points','price']][:15]
#scatter plot for wines with price>$150
sns.lmplot(x = 'price', y= 'points', fit_reg=False, data = df4)

def obtain_coutry_description(country):    
    '''obtain description list for the best rated wines (>87 points)
    Input
    country (str): country
    Return: description list
    '''
    mean_points_value =wine_groupped['points'].mean()
    remove_characters =re.
    description_list = list([wine_groupped['country']==country][wine_grouped['points']>mean_points_value]['description'])
    flat_list=[re.sub(remove_characters,' ',item) for sublist in description list for item in sublist]
    description=' '.join(flat_list)
    returndescription
def plot_country_wordcloud(words, country_mask, out_image_name):
    '''creates word cloud image

    Input
    words (list): list with description for country
    country_mask (str): path to jpg image
    out_image_name (str): path to output image

    Return: wordcloud object
    '''

    stopwords = ['now','nose','aroma', 'aromas', 'selection', 
'drink','wine','wines','come','add','give', 'come',
                 'flavor','flavors','one', 
'bottle','bottling','made','almost','note','notes','palate','finish',
                 'hint','hints','show','shows','open', 'now', 
'offering', 'almost', 'winery', 'making', 'lot', 
                 'along', 'offer', 'make', 'offers', 'yet']
    for w in stopwords:
        STOPWORDS.add(w)

    maska = np.array(Image.open(country_mask))
    wc = WordCloud(background_color="white", mask=maska,max_words=2000, random_state=42, stopwords=STOPWORDS, colormap = 'hsv')
    wc.generate(words)
    wc.to_file(out_image_name)

    return wc
#try it on US
US_mask = './Masks/united-states-silhouette.jpg'
us_description = obtain_coutry_description("US")
us_wc = plot_country_wordcloud(us_description, US_mask, './WineWordcloud/US.pdf')
us_wc.to_image()

#create dictionary of mask images
mask_images = {}
mask_images["US"] = './Masks/united-states-silhouette.jpg'
mask_images["France"] = './Masks/france-silhouette.jpg'
mask_images["Italy"] = './Masks/italy-silhouette.jpg'
mask_images["Spain"] = './Masks/spain-silhouette.jpg'
mask_images["South Africa"] = './Masks/south-africa-silhouette.jpg'
mask_images["New Zealand"] = './Masks/new-zealand-silhouette.jpg'
mask_images["Germany"] = './Masks/germany-silhouette.jpg'
mask_images["Argentina"] = './Masks/argentina-silhouette.jpg'
mask_images["Australia"] = './Masks/australia-silhouette.jpg'
mask_images["Austria"] = './Masks/austria-silhouette.jpg'
mask_images["Chile"] = './Masks/chile-silhouette.jpg'
mask_images["Portugal"] = './Masks/portugal-silhouette.jpg'

#create wine wordcloud images for 12 major countries
for country in major_countries:
  print(country)
  description = obtain_coutry_description(country)
  image_name = './WineWordcloud/' + '_'.join(country.split()) + '.pdf'
  print(image_name)
  wc = plot_country_wordcloud(description, mask_images[country], image_name )
  wc.to_image()

#creates scatter plot numerical features for 12 major countries with hue=country
df=wine_groupped[wine_groupped['country'].isin(major_countries)]
sns.set(style="ticks")
sns.pairplot(df,hue_order=major_countries,hue="country")