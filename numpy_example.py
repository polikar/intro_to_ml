import numpy as np
data = np.array([1,2,3,4,5])
weights = np.random.rand(5,1)

print(data.shape)
print(data.dtype)

print(weights.shape)
print(weights.dtype)

print(weights.T)
print(weights[0]) #first index
print(weights[1]) #second index
print(weights[1:3]) #2nd and 3rd index
print(weights[-1]) # last index

def weight_summary(weight_array):  #define a function to print the weight array
    for i in range(len(weight_array)):
        print("Weight Index ", i, ":", weight_array[i])


weight_summary(weights)

noise_magnitude = 0.1 * np.mean(data) # calculate the mean of the data
noise = np.random.normal(0, noise_magnitude, data.shape) # generate a random noise from normal distribution
noisy_data = data + noise_magnitude * noise  # Add noice to the data

print("Original Data: ", data)
print("Noisy Data: ", noisy_data)

#Analyze negative noise

negative_noise_mask = data > noisy_data  # where is data > noisy_data?
negative_data = noisy_data[negative_noise_mask]
negative_data = np.linalg.norm(negative_data, ord = 2)  #norm 2

print("Negative noise condition: ", negative_noise_mask)
print("Data with negative noise: ", negative_data)
print(f"Magnitide of negative noise: , {negative_noise_mask}")


# USe weights toget a weighted sum of asingle input of data
weights_transposed = weights.T
output = np.dot(weights_transposed, data) # dot product

print("Data shape:", data.shape)
print("Transposed weights shape:", weights_transposed.shape)
print("Output shape:", output.shape)

#Expand dimension

data = data.reshape(data.shape[0], 1)
output = np.dot(weights_transposed, data) # dot product of data and transposed weights

print("New data shape:", data.shape)
print("Output shape:", output.shape)

#Batch operation

data = np.random.rand(data.shape[0], 100)
output = np.dot(weights_transposed, data) # dot product of data and transposed

print("Batch data shape:", data.shape)
print("Batch output shape:", output.shape)

#Using linear algebra Package

#Calculuate the covariane matric on centered data

data_centered = data - np.mean(data, axis =0)
cov_matrix = np.cov(data_centered, rowvar=False)

#Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


#Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the top k eigenvectors (e.g., k=2)
k=2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

#Transform the original data 
principal_components = np.dot(data_centered, top_k_eigenvectors)

print("Principal components shape:", principal_components.shape)


# reading and writing arrays
#np.savetxt('array.txt', arr)
#arr_loaded = np.loadtxt('array.txt')






