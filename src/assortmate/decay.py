"""Functions for calcualting ancestry decay."""
import numpy as np
import numba


@numba.njit
def Zdiff_N01(ancestry_poll, N):
	"""Calculate gamma from Zaitlen et al. at distances up to N intervals.

	Gamma is the chance that two distinct sites share an ancestry.
	"""
	res = np.zeros(N, dtype=np.float64)
	res.fill(np.nan)
	res[0] = Zdiff01(ancestry_poll, ancestry_poll)

	Nx = np.min(np.array([N, len(ancestry_poll)]))

	# calculate at increasing offsets
	for i in range(1, Nx):
		a = ancestry_poll[:-i]
		b = ancestry_poll[i:]
		res[i] = Zdiff01(a, b)
	return(res)


@numba.njit
def Zdiff01(a, b):
	"""Calculate gamma from Zaitlen et al.

	Gamma is the chance that two distinct sites share an ancestry.
	@a is a vector of ancestry dosages at one site [0,1,2]
	@b is a vector of ancestry dosages at another site some distance away [0,1,2].

	a,b -> c [prob]
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
	anc_ind, intervals, func, all_other_pop):
	"""Return ancestry decay information in a way that is easier to bootstrap."""

	assert len(lefts) == len(rights)
	assert len(lefts_M) == len(rights_M)

	nind = len(inds) + len(all_other_pop)
	all_other_pop = set(all_other_pop)
	running = np.zeros((nind, len(intervals) + 1), dtype=np.float64)
	count = np.zeros((nind, len(intervals) + 1), dtype=np.float64)  # number of chromosomes evaluated

	# LAD within chromosomes
	for i in range(len(lefts)):  # for each chromosome
		left_bp = lefts[i]
		# right_bp = rights[i]
		rate = rates[i]
		left_M = lefts_M[i]
		right_M = rights_M[i]
		span = right_M - left_M

		# if the specified intervals extend past the chromosome edge, cut them short
		chrom_intervals = intervals[intervals < span]

		# units in bp
		ninterval = len(chrom_intervals)
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
				ancestry_poll = np.logical_and(
					ancestry_poll_points >= edges_left,
					ancestry_poll_points < edges_right).sum(1)
				autocov = func(ancestry_poll)
				if np.isnan(autocov[0]):
					assert False
				else:
					for j in range(ninterval):
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

	# combine with and across chromosomes
	running[:, -1] = across_running[:, 1:].sum(1)
	count[:, -1] = across_count[:, 1:].sum(1)

	return(running, count)


def get_ancestry_decay4(ts, genetic_map, target_pop, func, intervals):
	"""returns data in a form that allows a bootstrap"""

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
		intervals=intervals,
		func=func,
		all_other_pop=all_other_pop
	)
	return(running, count)
