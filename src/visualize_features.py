import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def visualize_1d(df):
    # Copy the DataFrame to avoid modifying the original
    df_to_norm = df.copy()

    # Separate the class labels
    classes = df_to_norm['Class Label']
    df_to_norm = df_to_norm.drop(columns=['Class Label'])

    # Apply standard scaling
    scale = MinMaxScaler().fit_transform(df_to_norm)

    # Create a new DataFrame with the scaled features
    norm_df = pd.DataFrame(scale, columns=df_to_norm.columns)

    # Insert the class labels back into the DataFrame
    norm_df.insert(0, 'Class Label', classes)
    
    print(norm_df)
    
    #PLOT FOR BEST FEATURE SELECTION USING FORWARD SELECTION (SMALL DATASET)
    # sns.scatterplot(data=norm_df,  x='Feature 9', y='Feature 9', hue='Class Label', palette='Set1')
    # plt.title("Highest Yielding Feature Subset (Forward Selection)")
    # plt.xlabel("Feature 3")
    # plt.ylabel("Feature 3")
    # plt.show()
    
    #PLOT FOR BEST FEATURE SELECTION USING BACKWARD SELECTION (SMALL DATASET)
    sns.scatterplot(data=norm_df,  x='Feature 9', y='Feature 9', hue='Class Label', palette='Set1')
    plt.title("Feature with Worst KNN Accuracy (Forward Selection)")
    plt.xlabel("Feature 3")
    plt.ylabel("Feature 3")
    plt.show()

def visualize_2d(df):

    # Copy the DataFrame to avoid modifying the original
    df_to_norm = df.copy()

    # Separate the class labels
    classes = df_to_norm['Class Label']
    df_to_norm = df_to_norm.drop(columns=['Class Label'])

    # Apply standard scaling
    scale = MinMaxScaler().fit_transform(df_to_norm)

    # Create a new DataFrame with the scaled features
    norm_df = pd.DataFrame(scale, columns=df_to_norm.columns)

    # Insert the class labels back into the DataFrame
    norm_df.insert(0, 'Class Label', classes)
    
    print(norm_df)

    sns.scatterplot(data=norm_df,  x='Feature 27', y='Feature 1', hue='Class Label', palette='Set1')
    plt.title("Features with Best KNN Accuracy (Forward Selection)")
    plt.xlabel("Feature 27")
    plt.ylabel("Feature 1")
    plt.show()

def high_dimensional(df):
    high_perf_df = df.copy()

    high_perf_df = high_perf_df.drop(columns=['Feature 9'])

    scaled_data = MinMaxScaler().fit_transform(high_perf_df)

    reduced_df = PCA(n_components=2).fit_transform(scaled_data)

    # Create the scatter plot
    sns.scatterplot(x=reduced_df[:, 0], y=reduced_df[:, 1], hue=high_perf_df['Class Label'], palette='Set1')

    # Set the labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title("PCA Graph of Best Feature(s) for Accuracy (Backward Elimination)")

    # Show the plot
    plt.show()



#df_small = create_df('data/CS170_Spring_2024_Small_data__16.txt')
df_large = create_df_large('data/CS170_Spring_2024_Large_data__16.txt')
df_small = create_df_small('data/CS170_Spring_2024_small_data__16.txt')
df_large_custom = create_df_large('data/large-test-dataset.txt')
df_small_custom = create_df_small('data/small-test-dataset.txt')

#visualize_1d(df_small)

#visualize_1d(df_small)
#high_dimensional(df_small)
# Copy the DataFrame to avoid modifying the original
df_to_norm = df_small.copy()

# Separate the class labels
classes = df_to_norm['Class Label']
df_to_norm = df_to_norm.drop(columns=['Class Label'])

# Apply standard scaling
scale = MinMaxScaler().fit_transform(df_to_norm)

# Create a new DataFrame with the scaled features
norm_df = pd.DataFrame(scale, columns=df_to_norm.columns)

# Insert the class labels back into the DataFrame
norm_df.insert(0, 'Class Label', classes)

sns.histplot(data=norm_df, x='Class Label', hue='Class Label', palette='Set1')
plt.show()

