import glob
import pandas as pd

seeds_path = "params/sim.jan13.seeds"
slim_path = "/home/users/waplesr/programs/slim/SLiM4/build/slim"


def make_targets():
	df = pd.read_csv(seeds_path, sep ='\t', header=None)
	df.columns = ['seed', 'G', 'A', 'R']
	targets = [f'results/{seed}.fit.npz' for seed in df.seed]
	return targets


def make_params_G(wildcards, output):
	df = pd.read_csv(seeds_path, sep ='\t', header=None)
	df.columns = ['seed', 'G', 'A', 'R']
	seed = int(wildcards.seed)
	return df.query('seed == @seed').G.values[0]


def make_params_A(wildcards, output):
	df = pd.read_csv(seeds_path, sep ='\t', header=None)
	df.columns = ['seed', 'G', 'A', 'R']
	seed = int(wildcards.seed)
	return df.query('seed == @seed').A.values[0]


def make_params_R(wildcards, output):
	df = pd.read_csv(seeds_path, sep ='\t', header=None)
	df.columns = ['seed', 'G', 'A', 'R']
	seed = int(wildcards.seed)
	return df.query('seed == @seed').R.values[0]


rule all:
	input: make_targets()


rule sim:
	# input: None,
	output: 'sims/{seed}.trees.tsz',
	params:
		trees='sims/{seed}.trees',
		G=make_params_G,
		A=make_params_A,
		R=make_params_R,
	shell:
		"""{slim_path} -d 'A=200' -d 'G={params.G}' -d 'nAdmix=5000' -d 'K=10000' -d 'M={params.A:.3f}' -d 'r_target={params.R:.3f}' -d 'nSample=1000' -d 'tsout="sims/{wildcards.seed}.trees"' -s {wildcards.seed} src/assortmate/assortative_mating.slim4"""

		"""

		tszip {params.trees}

		"""


rule tern:
	input: 'sims/{seed}.trees.tsz',
	output: 'results/{seed}.tern.npz',
	shell:
		"""python src/assortmate/calc_tern.py --ts_path {input} --output_path {output}"""


rule LAD:
	input: 'sims/{seed}.trees.tsz',
	output: 'results/{seed}.lad.npz',
	shell:
		"""python src/assortmate/calc_LAD.py --ts_path {input} --output_path {output}"""

rule fit:
	input:
		a = 'results/{seed}.lad.npz',
		b = 'results/{seed}.tern.npz',
	params:
		basename = 'results/{seed}'
	output:
		'results/{seed}.fit.npz',
	shell:
		"""python src/assortmate/calc_fit.py --basepath {params.basename} -n 1000"""
