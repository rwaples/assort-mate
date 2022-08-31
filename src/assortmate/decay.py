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

	# calcualte at increasing offsets
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
def calc_decay(
	lefts,
	rights,
	rates,
	lefts_M,
	rights_M,
	inds,
	anc_left,
	anc_right,
	anc_ind,
	interval_M,
	keep_interval,
	func):
	"""Calculate decay using the specified function across a data set of individuals and chromosomes."""
	assert len(lefts) == len(rights)
	assert len(lefts_M) == len(rights_M)

	# sum of probability of sharing ancestry
	running = np.zeros(keep_interval, dtype=np.float64)
	# number of chromosomes evaluated
	count = np.zeros(keep_interval, dtype=np.float64)

	# for each chromosome
	for i in range(len(lefts)):
		left_bp = lefts[i]
		# right_bp = rights[i]
		rate = rates[i]
		left_M = lefts_M[i]
		right_M = rights_M[i]
		ancestry_poll_points = (
			left_bp + np.arange(0, right_M - left_M, interval_M) / rate
		).astype(np.int64).reshape(-1, 1)
		# for each individual
		for ind in inds:
			edges_left = np.take(anc_left, np.where(anc_ind == ind)[0])
			edges_right = np.take(anc_right, np.where(anc_ind == ind)[0])
			ancestry_poll = np.logical_and(
				ancestry_poll_points >= edges_left, ancestry_poll_points < edges_right
			).sum(1)
			autocov = func(ancestry_poll, N=keep_interval)
			if np.isnan(autocov[0]):
				# didn't work, skip this chrom x ind, likely due to no ancestry switches
				pass
			else:
				for j in range(keep_interval - np.sum(np.isnan(autocov))):
					running[j] += autocov[j]
					count[j] += 1
	return(running, count)


def get_ancestry_decay_ts(ts, genetic_map, target_pop, func):
	"""Calculate ancestry decay from a tree sequence."""
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

	res = calc_decay(
		lefts=genetic_map.left[::2],
		rights=genetic_map.right[::2],
		rates=genetic_map.rate[::2],
		lefts_M=genetic_map.get_cumulative_mass(genetic_map.left[::2]),
		rights_M=genetic_map.get_cumulative_mass(genetic_map.right[::2]),
		inds=np.array(list(set(anc.ind))),
		anc_left=anc.left,
		anc_right=anc.right,
		anc_ind=anc.ind,
		interval_M=0.005,  # interval size in Morgans
		keep_interval=200,  # number of intervals to include
		func=func,
	)

	# decay is running / count
	running, count = res

	return running, count
