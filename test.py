import numpy as np
degree_data = np.load('Graph_Edge_Data/features_loop_7to10/adjacency_columns.npy')
print(f"Shape: {degree_data.shape}")
print(f"First 10 degree values: {degree_data[:10]}")