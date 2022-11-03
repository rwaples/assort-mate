import argparse
import pandas as pd
import numpy as np
import tskit
import tszip
import msprime
from funcs import Zdiff_N01, get_ancestry_decay, get_human_rec_map


def cli():
	parser = argparse.ArgumentParser(description="Calculate LAD")

	parser.add_argument(
		'-t', '--ts_path',
		type=str, required=True,
		help='Path to input tree sequence'
	)
	parser.add_argument(
		'-O', '--output_path',
		type=str, required=True,
		help='Path to output file'
	)
	parser.add_argument(
		'-M', '--map',
		type=str, required=False,
		help='Path to recombination map file'
	)
	parser.add_argument(
		'-I', '--cM_interval',
		type=float, default=0.5, required=False,
		help='LAD evaluation interval'
	)
	parser.add_argument(
		'-X', '--cM_max',
		type=int, default=100, required=False,
		help='max LAD reporting distance'
	)
	parser.add_argument(
		'-c', '--cores',
		type=int, required=False,
		help='Number of cores available, not yet implemented'
	)
	parser.add_argument(
		'-p', '--target_pop',
		type=int, default=1, required=False,
		help='Index of target population'
	)

	return parser.parse_args()


def main():
	args = cli()
	ts_path = args.ts_path
	if ts_path.endswith('.tsz'):
		ts = ts_path.decompress(ts_path)
	else:
		ts = tskit.load(ts_path)

	# load genetic map
	# rec_map = LOAD(args.map)
	genetic_map = get_human_rec_map()

	# calculate decay
	print('__Calculating LAD__')
	running, count = get_ancestry_decay(
		ts=ts,
		genetic_map=genetic_map,
		target_pop=args.target_pop,
		func=Zdiff_N01,
		cM_interval=args.cM_interval,
		cM_max=args.cM_max
	)
	# print(running.shape, count.shape)

	# write decay files
	np.savez_compressed(args.output_path, running=running, count=count)


if __name__ == '__main__':
	main()
