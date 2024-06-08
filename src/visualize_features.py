import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Converts data in small txt file to dataframe
def create_df_small(file):
    file_path = file
    
    df = pd.DataFrame()

    with open(file_path, 'r') as dataset:
        data = dataset.read()

    elements = [float(num) for num in data.split()]

    features = 11

    for i in range(0, len(elements), features):
        line = elements[i:i + features]

        df = pd.concat([df, pd.DataFrame([line])], ignore_index=True)
    
    new_cols = []
    for i in range(df.shape[1]):
        if (i == 0):
            new_cols.append(f'Class Label')
        else:
            new_cols.append(f'Feature {i}')

    df.columns = new_cols

    return df

#Converts data in large txt file to dataframe
def create_df_large(file):
    file_path = file
    
    df = pd.DataFrame()

    with open(file_path, 'r') as dataset:
        data = dataset.read()

    elements = [float(num) for num in data.split()]
    features = 40

    for i in range(0, len(elements), features + 1):
        line = elements[i:i + features + 1]
        df = pd.concat([df, pd.DataFrame([line])], ignore_index=True)

    new_cols = []
    for i in range(df.shape[1]):
        if (i == 0):
            new_cols.append(f'Class Label')
        else:
            new_cols.append(f'Feature {i}')

    df.columns = new_cols

    return df

#Create visuals for feature subsets that only have one feature
def visualize_1d(df):
    df_to_norm = df.copy()

    #Separate the class labels
    # classes = df_to_norm['Class Label']
    # df_to_norm = df_to_norm.drop(columns=['Class Label'])

    # #Apply MinMax Scaling
    # scale = MinMaxScaler().fit_transform(df_to_norm)

    # norm_df = pd.DataFrame(scale, columns=df_to_norm.columns)

    # #Insert the class labels back into the DataFrame
    # norm_df.insert(0, 'Class Label', classes)
    
    # print(norm_df)
    
    # #PLOT FOR BEST FEATURE SELECTION USING BACKWARD SELECTION (SMALL DATASET)
    # sns.scatterplot(data=norm_df,  x='Feature 9', y='Feature 9', hue='Class Label', palette='Set1')
    # plt.title("Feature with Worst KNN Accuracy (Forward Selection)")
    # plt.xlabel("Feature 3")
    # plt.ylabel("Feature 3")
    # plt.show()

    sns.countplot(data=df_to_norm, x='Class Label')
    plt.title("Count of Data Objects Between Classes (Small Dataset)")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.show()


#Create visuals for feature subsets that only have two feature
def visualize_2d(df):

    df_to_norm = df.copy()

    #Separate the class labels
    classes = df_to_norm['Class Label']
    df_to_norm = df_to_norm.drop(columns=['Class Label'])

    scale = MinMaxScaler().fit_transform(df_to_norm)

    norm_df = pd.DataFrame(scale, columns=df_to_norm.columns)

    norm_df.insert(0, 'Class Label', classes)

    sns.scatterplot(data=norm_df,  x='Feature 5', y='Feature 2', hue='Class Label', palette='Set1')
    plt.title("Correlation Analysis of Feature 5 and 2 (Small Dataset)")
    plt.xlabel("Feature 5")
    plt.ylabel("Feature 2")
    plt.show()

#Create visuals for feature subsets that many features (used only for higher dimensional data)
def high_dimensional(df):
    high_perf_df = df.copy()

    scaled_data = MinMaxScaler().fit_transform(high_perf_df)

    reduced_df = PCA(n_components=2).fit_transform(scaled_data)

    sns.scatterplot(x=reduced_df[:, 0], y=reduced_df[:, 1], hue=high_perf_df['Class Label'], palette='Set1')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title("PCA Graph of Best Feature(s) for Accuracy (Backward Elimination)")

    plt.show()


#creates dataframes for each of the datasets
df_large = create_df_large('data/CS170_Spring_2024_Large_data__16.txt')
df_small = create_df_small('data/CS170_Spring_2024_small_data__16.txt')
df_large_custom = create_df_large('data/large-test-dataset.txt')
df_small_custom = create_df_small('data/small-test-dataset.txt')



#######################################################Code Below is used simply for fast access to visuals#######################


# visualize_1d(df_small)
visualize_2d(df_small_custom)
# high_dimensional(df_large)

# K_vals = [2, 3, 4, 5]
# Accs = [.96, .95, .93, .91]

# sns.lineplot(x=K_vals, y=Accs)
# plt.xlabel("Values of K")
# plt.ylabel("Accuracy")
# plt.title("Values of K and Accuracy")
# plt.show()

