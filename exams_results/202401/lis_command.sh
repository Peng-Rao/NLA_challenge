# compute the largest eigenvalue
echo "******************************************************************************"
mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e pi -etol 1.e-7 -emaxiter 50000

# Find a shift
echo "******************************************************************************"
mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e pi -etol 1.e-7 -emaxiter 50000 -shift 7

# compute the smallest eigenvalue
echo "******************************************************************************"
mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e ii -etol 1.e-7 -emaxiter 50000