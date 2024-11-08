mpirun -n 4 ./eigen1 gr_30_30.mtx eigvec.txt hist.txt -e pi -etol 1.e-8 -emaxiter 5000

mpirun -n 4 ./eigen1 gr_30_30.mtx eigvec.txt hist.txt -e ii -etol 1.e-8 -emaxiter 5000

mpirun -n 4 ./eigen2 gr_30_30.mtx evals.mtx eigvecs.mtx res.txt iters.txt -ss 8 -e si -p jacobi -etol 1.e-10 -emaxiter 5000