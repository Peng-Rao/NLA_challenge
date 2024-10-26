# Using the proper iterative solver available in the LIS library compute the largest eigenvalue of A^T A up to a tolerance of 10−8.
# Report the computed eigenvalue.
# Is the result in agreement with the one obtained in the previous point?

# calculate the largest eigenvalue of A^T A
mpirun -n 8 ./extern/lis/bin/eigen1 ./ch2_result/gram_matrix.mtx  eigvec.txt hist.txt -e pi -etol 1.e-8

# calculate the largest eigenvalue of A^T A, using inverse power method
mpirun -n 8 ./extern/lis/bin/eigen1 ./ch2_result/gram_matrix.mtx  eigvec.txt hist.txt -e ii -etol 1.e-8 -shift 1e9

# calculate the smallest eigenvalue of A^T A, using inverse power method
mpirun -n 8 ./extern/lis/bin/eigen1 ./ch2_result/gram_matrix.mtx  eigvec.txt hist.txt -e ii -etol 1.e-8 -shift 0
