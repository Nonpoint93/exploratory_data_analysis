from utils import db_connect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# your code here

def delete_irrelevant_information(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.drop(['id', 'host_id', 'name', 'host_name', 'availability_365', 'calculated_host_listings_count', 'neighbourhood', 'latitude', 'longitude', 'last_review'], axis=1)
    return dataframe

def delete_duplicate_values(dataframe: pd.DataFrame) -> pd.DataFrame:

    if dataframe['name'].duplicated().sum() or dataframe['host_id'].duplicated().sum() or dataframe['id'].duplicated().sum():
        dataframe = dataframe.drop_duplicates()
        print(dataframe.info())
        print(dataframe.head())
    return dataframe

def display_analysis_categorical(dataframe):
    """ Análisis de variables categóricas """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(ax=ax1, data=dataframe, x='room_type', hue='room_type', multiple='stack', shrink=.8, palette='viridis')
    sns.histplot(ax=ax2, data=dataframe, x='neighbourhood_group', hue='neighbourhood_group', multiple='stack', shrink=.8, palette='viridis').set_ylabel(None)
    plt.tight_layout()
    plt.show()

    pd.crosstab(dataframe['room_type'], dataframe['neighbourhood_group'], normalize='index').plot(kind='bar', stacked=True)
    plt.title("Room Type by Neighbourhood Group")
    plt.show()

def display_analysis_numerical_values(dataframe: pd.DataFrame):
    """ Análisis de variables numéricas """
    _, axis = plt.subplots(2, 2, figsize = (10, 7))
    sns.histplot(ax = axis[0,0], data = dataframe, x = "price").set_ylabel(None)
    sns.boxplot(ax = axis[1,0], data = dataframe, x = "minimum_nights").set_ylabel(None)
    sns.histplot(ax = axis[0,1], data = dataframe, x = "number_of_reviews").set_ylabel(None)
    sns.boxplot(ax = axis[1,1], data = dataframe, x = "reviews_per_month").set_ylabel(None)
    plt.tight_layout()
    plt.show()

    sns.scatterplot(data=dataframe, x='price', y='minimum_nights')
    plt.title("Price vs Minimum Nights")
    plt.show()

def display_analysis_correlation(dataframe):
    """ Análisis de correlación """
    _, axis = plt.subplots(1, 5, figsize = (15, 12))
    sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=axis[0])
    sns.scatterplot(data=dataframe, x='number_of_reviews', y='reviews_per_month', ax=axis[1])
    sns.scatterplot(data=dataframe, x='minimum_nights', y='price', ax=axis[2])
    sns.boxplot(data=dataframe, x='room_type_n', y='price', ax=axis[3])
    sns.scatterplot(data=dataframe, x='price', y='minimum_nights', hue='room_type', ax=axis[4])
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def remove_outliers(dataframe, column) -> pd.DataFrame:
    """ Eliminar outliers """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]



def main():
    
    dataframe = pd.read_csv('data/raw/AB_NYC_2019.csv', delimiter=',')
    print(dataframe.shape)
    print(dataframe.info())
    print(dataframe.describe())
    dataframe = delete_duplicate_values(dataframe)
    dataframe.loc[dataframe['number_of_reviews']==0, 'reviews_per_month'] = dataframe.loc[dataframe['number_of_reviews'] == 0, 'reviews_per_month'].fillna(0)
    dataframe['room_type_n'] = pd.factorize(dataframe['room_type'])[0]
    dataframe = remove_outliers(dataframe=dataframe, column="price")
    dataframe = remove_outliers(dataframe=dataframe, column="minimum_nights")
    dataframe = dataframe[dataframe['minimum_nights'] <= 15 ]
    dataframe = dataframe[dataframe['price'] > 0]

    dataframe = delete_irrelevant_information(dataframe)

    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)

    train_df.to_csv('./data/processed/train_data.csv', index=False)
    test_df.to_csv('./data/processed/test_data.csv', index=False)

    display_analysis_categorical(dataframe=dataframe)
    display_analysis_numerical_values(dataframe=dataframe)
    display_analysis_correlation(dataframe)


if __name__ == '__main__':
    main()