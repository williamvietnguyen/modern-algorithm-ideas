# William Nguyen
# email: williamvnguyen2@gmail.com

import matplotlib.pyplot as plt
import numpy as np
from pca import PCA


def build_data(data):
    def mode(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    # Calculate the most frequent nucleobase for each column
    most_frequent_nuclobase = []
    for j in range(data.shape[1]):
        current = mode(data[:, j])
        most_frequent_nuclobase.append(current) 
    # Calculate the binary matrix
    # 1 if this person has the column-j mode nucleobase in column j
    # 0 otherwise
    X = np.ones(data.shape)
    for j in range(data.shape[1]):
        current_mode = most_frequent_nuclobase[j]
        for i in range(data.shape[0]):
            if data[i, j] == current_mode:
                X[i, j] = 0 
    return X

def population_to_indices(information):
    populations = information[:,2]
    # Create a dictionary that maps each population to a list of indices (aka rows) from the original table
    population_indices = {}
    for i in range(len(populations)):
        current = populations[i]
        if current not in population_indices:
            population_indices[current] = []
        population_indices[current].append(i)
    return population_indices

if __name__ == '__main__':
    data = np.genfromtxt('data/genome-data.txt', dtype=str, delimiter=' ')
    information, data = data[:, :3], data[:, 3:]
    X = build_data(data)
    population_indices = population_to_indices(information)
    model = PCA()
    X_transformed = model.fit_transform(X, 2)

    # Plot the data for each population
    plt.figure(figsize=(10,7))
    for population in population_indices:
        indices = population_indices[population]
        plt.scatter(X_transformed[indices, 0], X_transformed[indices, 1])
    plt.legend(population_indices.keys())
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Genome data from different populations projected onto the first 2 PC')
    plt.savefig('./plots/pc1_vs_pc2.png')


