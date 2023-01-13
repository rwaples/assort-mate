import os
import numpy as np
import tskit
import tszip

niter = 5000

for i in range(niter):
	G = np.random.randint(2, 30)
	M = np.random.uniform(0.1, 0.9)
	R = np.random.uniform(0, 1)
	seed = np.random.randint(2, 9000000)
	path = f"""/home/users/waplesr/programs/slim/SLiM4/build/slim -d 'A=200' -d 'G={G}' -d 'nAdmix=5000' -d 'K=10000' -d 'M={M:.3f}' -d 'r_target={R:.3f}' -d 'nSample=200' -d 'tsout="/home/users/waplesr/assort-mate/sims/{seed}.trees"' -s {seed} /home/users/waplesr/assort-mate/src/assortmate/assortative_mating.slim4"""
	os.system(path)
	with open('sim.log', 'a') as OUTFILE:
		OUTFILE.write(f"{seed}\t{G}\t{M}\t{R}\n")
	ts = tskit.load(f'/home/users/waplesr/assort-mate/sims/{seed}.trees')
	tszip.compress(ts, f'/home/users/waplesr/assort-mate/sims/{seed}.tsz')
	os.remove(f'/home/users/waplesr/assort-mate/sims/{seed}.trees')
