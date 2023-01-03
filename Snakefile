import glob
def make_targets():
	ins = glob.glob('sims/*.tsz')
	targets  = [f.replace('sims', 'results').replace('tsz', 'fit.npz') for f in ins]
	return targets



rule all:
	input: make_targets()


rule tern:
	input: 'sims/{seed}.tsz',
	output: 'results/{seed}.tern.npz',
	shell:
		"""python src/assortmate/calc_tern.py --ts_path {input} --output_path {output}"""


rule LAD:
	input: 'sims/{seed}.tsz',
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
