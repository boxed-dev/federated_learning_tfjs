import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Flatten the images
# x_train = x_train.reshape(-1, 28*28)
# x_test = x_test.reshape(-1, 28*28)

# # Create a DataFrame
# train_df = pd.DataFrame(x_train)
# train_df['label'] = y_train

# # Split into two subsets
# client1_df = train_df[train_df['label'] < 5]
# client2_df = train_df[train_df['label'] >= 5]

# # Save to CSV
# client1_df.to_csv('../client/mnist_data/client1.csv', index=False)
# client2_df.to_csv('../client/mnist_data/client2.csv', index=False)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images
x_test = x_test.reshape(-1, 28*28)

# Create a DataFrame
test_df = pd.DataFrame(x_test)
test_df['label'] = y_test

# Save to CSV
test_df.to_csv('../client/mnist_data/test.csv', index=False)