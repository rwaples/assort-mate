"""Functions for calculating ancestry decay."""
import numpy as np
import numba
import stdpopsim
import msprime


@numba.njit
def Zdiff_phased_outer(ancestry_poll):
	"""Return gamma from Zaitlen et al.  The chance that two distinct sites share an ancestry.
	@a is a vector of ancestries dosages at a series of sites, values in [0,1,2]
	@b is a vector of ancestries dosages at a series of sites some distance from
		the sites in @a, values also in [0,1,2],
		.
	"""
	N = len(ancestry_poll)
	res = np.zeros(N, dtype=np.float64)
	res.fill(np.nan)

	# at a distance of zero
	res[0] = Zdiff_phased_inner(ancestry_poll, ancestry_poll)

	for i in range(1, N):
		a = ancestry_poll[:-i]
		b = ancestry_poll[i:]
		res[i] = Zdiff_phased_inner(a, b)
	return(res)


@numba.njit
def Zdiff_phased_inner(a, b):
	"""Return gamma_11 + gamma_22 from Zaitlen et al.
	The chance that two distinct sites share an ancestry.
	@a is a vector of ancestry dosages at a series of sites, values in [0,1,2]
	@b is a vector of ancestry dosages at a series of sites [0,1,2], some distance from the sites in a

	(dosage at first site), (dosage at second site) -> (index into res_possible) ([gamma])
	0,0 -> 0 [1.0]
	1,0 -> 1 [0.5]
	2,0 -> 2 [0.0]
	0,1 -> 3 [0.5]
	1,1 -> 4 [0.5]
	2,1 -> 5 [0.5]
	0,2 -> 6 [0.0]
	1,2 -> 7 [0.5]
	2,2 -> 8 [1.0]

	"""
	res_possible = np.array([1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 1.0])
	c = a + 3 * b

	return np.nanmean(res_possible[c])


@numba.njit
def calc_decay_boot(
	lefts, rights, rates, lefts_M,
	rights_M, inds, anc_left, anc_right,
	anc_ind, func, all_other_pop,
	cM_interval, cM_max,
	include_cross_chrom=True):
	"""Return ancestry decay information in a way that is easier to bootstrap."""

	assert len(lefts) == len(rights)
	assert len(lefts_M) == len(rights_M)

	nind = len(inds) + len(all_other_pop)
	all_other_pop = set(all_other_pop)
	nkeep = len(np.arange(0, cM_max / 100, cM_interval / 100))  # convert cM to M

	running = np.zeros((nind, nkeep + 1), dtype=np.float64)
	# number of chromosomes evaluated
	count = np.zeros((nind, nkeep + 1), dtype=np.float64)

	# LAD within chromosomes
	for i in range(len(lefts)):  # for each chromosome
		left_bp = lefts[i]
		# right_bp = rights[i]
		rate = rates[i]
		left_M = lefts_M[i]
		right_M = rights_M[i]
		span = right_M - left_M

		# if the specified intervals extend past the chromosome edge, cut them short
		chrom_intervals = np.arange(0, span, cM_interval / 100)
		ninterval = len(chrom_intervals)
		# units in bp

		ancestry_poll_points = (left_bp + chrom_intervals / rate).astype(np.int64).reshape(-1, 1)
		k = 0
		for ind in inds:
			if ind in all_other_pop:
				for j in range(ninterval):
					running[k, j] += 1
					count[k, j] += 1
			else:
				edges_left = np.take(anc_left, np.where(anc_ind == ind)[0])
				edges_right = np.take(anc_right, np.where(anc_ind == ind)[0])
				# diploid ancestry at each polled point
				ancestry_poll = np.logical_and(
					ancestry_poll_points >= edges_left,
					ancestry_poll_points < edges_right).sum(1)
				autocov = func(ancestry_poll)
				if np.isnan(autocov[0]):
					assert False
				else:
					for j in range(min(nkeep, ninterval)):
						running[k, j] += autocov[j]
						count[k, j] += np.isfinite(autocov[j])
			k += 1

	# LAD across chromosomes
	midpoints = (lefts + rights) / 2
	midpoints = midpoints.astype(np.int64).reshape(-1, 1)
	NK = len(midpoints)
	across_running = np.zeros((nind, NK), dtype=np.float64)
	across_count = np.zeros((nind, NK), dtype=np.float64)  # number of chromosomes evaluated

	k = 0
	for ind in inds:
		edges_left = np.take(anc_left, np.where(anc_ind == ind)[0])
		edges_right = np.take(anc_right, np.where(anc_ind == ind)[0])
		ancestry_poll = np.logical_and(midpoints >= edges_left, midpoints < edges_right).sum(1)
		autocov = func(ancestry_poll)
		for j in range(NK):
			across_running[k, j] += autocov[j]
			across_count[k, j] += 1
		k += 1

	# combine within and across chromosomes
	running[:, -1] = across_running[:, 1:].sum(1)
	count[:, -1] = across_count[:, 1:].sum(1)

	return running, count


def get_ancestry_decay(ts, genetic_map, target_pop, func, cM_interval, cM_max):
	"""Return LAD data in a form that allows a bootstrap"""

	# setup
	max_node_age = ts.tables.nodes.asdict()['time'].max()

	anc = ts.tables.map_ancestors(
		samples=ts.samples(),
		ancestors=np.where(
			(ts.tables.nodes.asdict()['population'] == target_pop) &
			(ts.tables.nodes.asdict()['time'] == max_node_age)
		)[0]
	)
	# ind of each child sample
	Nsamp = len(ts.samples())
	Nind = np.int64(Nsamp / 2)
	ind_of_sample = dict(zip(np.arange(Nsamp), np.arange(Nind).repeat(2)))
	anc.ind = np.vectorize(ind_of_sample.__getitem__)(anc.child)

	# inds that do not have ancestry from the target population
	all_other_pop = np.array(list(set(range(Nind)) - set(anc.ind)), dtype='int64')

	# calculation
	running, count = calc_decay_boot(
		lefts=genetic_map.left[::2],
		rights=genetic_map.right[::2],
		rates=genetic_map.rate[::2],
		lefts_M=genetic_map.get_cumulative_mass(genetic_map.left[::2]),
		rights_M=genetic_map.get_cumulative_mass(genetic_map.right[::2]),
		inds=np.array(list(set(anc.ind))),
		anc_left=anc.left,
		anc_right=anc.right,
		anc_ind=anc.ind,
		cM_interval=cM_interval,
		cM_max=cM_max,
		func=func,
		all_other_pop=all_other_pop
	)
	return running, count


def get_ternary(ts, ref_pop):
	"""Compute the ternary ancestry fractions for each individual from a ts.

	ternary = ternary ancestry fractions (N, 3) array of floats
	N = number of indiviudals
	assumes just two admixing populations
	ts = tree-sequence
	"""
	L = ts.sequence_length
	max_node_age = ts.tables.nodes.asdict()['time'].max()

	# match each interval in the samples to an ind from an ancestral population
	anc = ts.tables.map_ancestors(
		samples=ts.samples(),
		ancestors=np.where(
			(ts.tables.nodes.asdict()['population'] == ref_pop)
			& (ts.tables.nodes.asdict()['time'] == max_node_age)
		)[0]
	)

	# ancestry of each interval
	pop_of_node = dict()
	for node in ts.nodes():
		pop_of_node[node.id] = node.population
	anc.ancestry = np.vectorize(pop_of_node.__getitem__)(anc.parent)

	# ind of each child (sample)
	Nsamp = len(ts.samples())
	Nind = int(Nsamp / 2)
	ind_of_sample = dict(zip(np.arange(Nsamp), np.arange(int(Nsamp / 2)).repeat(2)))
	anc.ind = np.vectorize(ind_of_sample.__getitem__)(anc.child)

	# compute the ternary fractions
	ternary = np.zeros([Nind, 3], dtype='float64')
	for i, ind in enumerate(range(Nind)):
		# get the unique ancestry switch points for the individual
		lefts = np.take(anc.left, np.where(anc.ind == ind))
		rights = np.take(anc.right, np.where(anc.ind == ind))
		endpoints = np.unique(np.concatenate([lefts, rights]))
		# and the length of each ancestry segment
		span = np.diff(endpoints)
		#  a point that should be inside each interval
		midpoints = endpoints[1:] - 1

		# for each midpoint how many intervals it is inside?
		inside_n = np.logical_and(
			midpoints.reshape(-1, 1) > lefts,
			midpoints.reshape(-1, 1) < rights
		).sum(1)
		# add up the intervals that contribute to each
		frac_pop1pop1 = span[np.where(inside_n == 2)].sum() / L
		frac_pop1pop2 = span[np.where(inside_n == 1)].sum() / L
		frac_pop2pop2 = 1 - (frac_pop1pop1 + frac_pop1pop2)
		ternary[i] = (frac_pop1pop1, frac_pop1pop2, frac_pop2pop2)

	return ternary


def get_human_rec_map(print_notice=False):
	"""Return a (discrete) recombination map for 22 human autosomes.

	There is a 1 bp region of 0.5 recombination rate between each chromosome.
	"""
	map_of_chr = {}
	species = stdpopsim.get_species('HomSap')
	for contig in [f'chr{x}' for x in range(1, 23)]:
		map_of_chr[contig] = species.get_contig(contig).recombination_map

	pos_list = []
	rates_list = []
	# shift the positions on each chromosome due to concatenation of the genome
	shifts = [0]
	for i in range(1, 23):
		chrom = f'chr{i}'
		# update to 0.2 API
		# pos = map_of_chr[chrom].get_positions()
		# rates = map_of_chr[chrom].get_rates()
		pos = map_of_chr[chrom].position.copy()
		rates = map_of_chr[chrom].rate.copy()
		# rates[-1] = .5
		rates = [rates[0], 0.5]
		rates_list.extend(rates)
		if i > 1:
			shift = pos_list[-1]
			pos = [x + 1 + shift for x in pos]
			pos_list.extend(pos)
			shifts.append(shift)
		else:
			pos_list.extend(pos)

	# print(pos_list)
	# print(rates_list)
	human_map = msprime.RateMap(
		position=pos_list,
		rate=rates_list[:-1],
	)
	bp = human_map.sequence_length
	M = human_map.total_mass

	if print_notice:
		print('''human recombination map
			sequence length (bp) {bp}
			total recombination rate (M):: {M:0.4}'''.format(bp=int(bp), M=M))

	return human_map
