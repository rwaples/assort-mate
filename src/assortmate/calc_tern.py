import numpy as np
import tskit
import tszip
import argparse
from funcs import get_ternary


def cli():
	parser = argparse.ArgumentParser(description="Calculate Q")

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
	return parser.parse_args()


def main():
	args = cli()
	ts_path = args.ts_path
	if ts_path.endswith('.tsz'):
		ts = tszip.decompress(ts_path)
	else:
		ts = tskit.load(ts_path)

	print('__Calculating Q__')
	print(len(ts.populations()))
	tern = get_ternary(ts, ref_pop=1)

	# for line in tern:
	# 	print(line)
	print(f'Saving to {args.output_path}')
	np.savez_compressed(args.output_path, tern=tern)


if __name__ == '__main__':
	main()
