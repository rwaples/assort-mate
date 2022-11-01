import argparse
import pandas as pd
import tskit
import tszip

from decay_funcs import Zdiff_N01, get_ancestry_decay

# import msprime


def cli():
	parser = argparse.ArgumentParser(description="Calculate LAD")

	parser.add_argument(
		'-t', '--ts_path',
		type=str, required=True,
		help='Path to input tree sequence'
	)
	parser.add_argument(
		'-M', '--map',
		type=str, required=True,
		help='Path to recombination map file'
	)
	parser.add_argument(
		'-O', '--output_path',
		type=str, required=True,
		help='Path to output file'
	)
	parser.add_argument(
		'-I', '--intervals',
		type=str, required=False,
		help='Path to interval file'
	)
	parser.add_argument(
		'-c', '--cores',
		type=int, required=False,
		help='Number of cores available'
	)
	parser.add_argument(
		'-p', '--target_pop',
		type=int, default=0, required=False,
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
	rec_map = LOAD(args.map)
	# load intervals
	if arg.intervals:
		intervals = LOAD(arg.intervals)
	else:
		pass
		# calculate intervals

	# calculate decay
	running, count = get_ancestry_decay(
		ts=ts,
		genetic_map=rec_map,
		target_pop=args.target_pop,
		func=Zdiff_N01,
		intervals=intervals
	)



	# write output
