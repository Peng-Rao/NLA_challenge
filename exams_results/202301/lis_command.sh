# solve linear system Ax=b
echo "******************************************************************************"
mpirun -n 4 ./test1 A_1.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10

echo "******************************************************************************"
mpirun -n 4 ./test1 A_1.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p hybrid

# ILU preconditioner
echo "******************************************************************************"
mpirun -n 4 ./test1 A_1.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p ilu

echo "******************************************************************************"
mpirun -n 4 ./test1 A_1.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p ilu -ilu_fill 2

echo "******************************************************************************"
mpirun -n 4 ./test1 A_1.mtx 2 sol.txt hist.txt -i bicgstab -tol 1e-10 -p ilu -ilu_fill 2 -adds true -adds_iter 3