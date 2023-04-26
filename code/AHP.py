import numpy as np

matrix = np.array([[1,4/3,4/1,4/2,4/5],
                   [3/4,1,3/1,3/2,3/5],
                   [1/4,1/3,1,1/2,1/5],
                   [2/4,2/3,2/1,1,2/5],
                   [5/4,5/3,5/1,5/2,1]])

eigenvalues, eigenvectors = np.linalg.eig(matrix)
max_eigenvalue_index = np.argmax(eigenvalues)
weights = np.real(eigenvectors[:, max_eigenvalue_index]) /\
          np.sum(np.real(eigenvectors[:, max_eigenvalue_index]))

factors = ["PD", "LT", "AQI", "GDP", "UR"]
for i in range(5):
    print(f"{factors[i]} ：{weights[i]:.2f}")

ci = (max(eigenvalues)-5) / 4
cr = ci / 1.12
if cr < 0.1:
    print("Consistency passes with high confidence")
else:
    print("Consistency does not pass, matrix needs to be readjusted")