# solve eigenvalues
echo "******************************************************************************"
mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e pi -etol 1.e-10

# Inverse power method
mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e ii -etol 1.e-10 -i cg -shift 50.0

# rayleigh quotient method
mpirun -n 4 ./eigen1 exer2.mtx eigvec.txt hist.txt -e rqi -etol 1.e-10 -i cg

mpirun -n 4 ./eigen1 exer2.mtx eigvec.mtx hist.txt -e ii -etol 1.e-10 -i cg -p ssor

mpirun -n 4 ./eigen1 exer2.mtx eigvec.mtx hist.txt -e ii -etol 1.e-10 -i jacobi -p jacobi