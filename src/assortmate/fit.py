"""Functions for fitting ancestry decay and ancestry heterozygosity."""

import numba
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('/home/kele/Documents/abc_dev/abcADMIX/abc/project'))
import sum_stats
import rec_map

LAD0 = None
alpha = None
beta = None
R_est2 = None

genetic_map = rec_map.get_human_rec_map()


@numba.njit
def Zfunc(c, intercept, R, G):
	"""
	Use to fit ancestry decay.

	The variables alpha, beta, and LAD0 should be defined outside the function.

	c -- recombination fraction
	intercept -- the intercept
	R -- correlation in ancestry between mates
	G -- number of generations since admixture
	"""
	p0 = R * LAD0

	LAD = (
		intercept +
		alpha**2 +
		beta**2 +
		(1 - c)**G * LAD0 +
		c * p0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))
	)
	return LAD


@numba.njit
def next_fk(f, k, R):
	"""Estimate the next f and k values.

	See Fig 1 in Crow and Felesenstein (1968).

	f -- "correlation in value of homologous genes"
	k -- "correlation of nonhomologous genes in the same gamete""
	R -- correlation in ancestry between mates
	"""
	n = 10000  # "number of gene loci that determine the trait"
	c = 0.5  # recombination fraction
	f1 = R / (2 * n) * (1 + n * f + (n - 1) * k)
	k1 = (1 - c) * k + c * f
	return f1, k1, R


@numba.njit
def xnext_fk(G, r):
	# intial values
	G = G + 1  # adjust the generations to match the two methods
	f = 1
	k = 1
	# this weighting provides a linear interpolation between generations
	Gi = int(np.floor(G))
	weight = G - Gi
	for i in range(Gi):
		f, k, r = next_fk(f, k, r)
	# one more for interpolation
	fi, ki, r = next_fk(f, k, r)
	f = f * (1 - weight) + fi * weight
	return f


@numba.njit
def step1(c, intercept, G):
	"""Use the estimate of R from step 2.

	c -- recombination fraction
	intercept -- the intercept
	G -- number of generations since admixture
	"""
	R = R_est2
	p0 = R * LAD0
	return intercept + alpha**2 + beta**2 + (1 - c)**G * LAD0 + c * p0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))


@numba.njit
def step2(G, R):
	"""Uses the estimate of G from step 1

	G -- number of generations since admixture
	R -- correlation in ancestry between mates
	"""
	G = G + 1
	f = 1
	k = 1
	# this weighting provides a linear interpolation between generations
	Gi = int(np.floor(G))
	weight = G - Gi
	for i in range(Gi):
		f, k, r = next_fk(f, k, R)
	# one more for interpolation
	fi, ki, r = next_fk(f, k, R)
	f = f * (1 - weight) + fi * weight
	return f


def Haldane(M):
	"""Convert from Morgans to recombination fraction with the Haldane mapping function.

	M -- numpy array of positions given in Morgans
	"""
	c = 0.5 * (1 - np.exp(-2 * M))
	return c


def inv_Haldane(c):
	"""Convert from recombination fraction to Morgans with the inverse Haldane mapping function.

	c -- numpy array of positions given in recombination fractions.
	"""
	M = np.abs(np.log(1 - 2 * c) / - 2)
	return M


def generate_query_points(max_c, interval):
	"""Generate an array of points (in Morgans) to query for LAD

	max_c -- maximum recombination fraction between sites to query.
	interval -- interval between adjacent query points (in units of recombination fraction)
	"""
	query_M = inv_Haldane(np.arange(0, max_c + interval, interval))
	return query_M


@numba.njit
def calc_decay_boot(
	lefts, rights, rates, lefts_M,
	rights_M, inds, anc_left, anc_right,
	anc_ind, interval_M, keep_interval, func, all_other_pop):
	"""Return ancestry decay information in a way that is easier to bootstrap."""

	assert len(lefts) == len(rights)
	assert len(lefts_M) == len(rights_M)

	# add another dimension to store ind-specific data
	nind = len(inds) + len(all_other_pop)
	all_other_pop = set(all_other_pop)
	running = np.zeros((nind, keep_interval), dtype=np.float64)
	count = np.zeros((nind, keep_interval), dtype=np.float64)  # number of chromosomes evaluated
	for i in range(len(lefts)):  # for each chromosome
		left_bp = lefts[i]
		# right_bp = rights[i]
		rate = rates[i]
		left_M = lefts_M[i]
		right_M = rights_M[i]
		ancestry_poll_points = (left_bp + np.arange(0, right_M - left_M, interval_M)/rate).astype(np.int64).reshape(-1, 1)
		k = 0
		for ind in inds:
			if ind in all_other_pop:
				for j in range(len(ancestry_poll_points)):
					running[k, j] += 1
					count[k, j] += 1
			else:
				edges_left = np.take(anc_left, np.where(anc_ind == ind)[0])
				edges_right = np.take(anc_right, np.where(anc_ind == ind)[0])
				ancestry_poll = np.logical_and(
					ancestry_poll_points >= edges_left,
					ancestry_poll_points < edges_right
				).sum(1)
				autocov = func(ancestry_poll, N=keep_interval)
			if np.isnan(autocov[0]):
				assert False
			else:
				for j in range(keep_interval - np.sum(np.isnan(autocov))):
					running[k, j] += autocov[j]
					count[k, j] += 1
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
		autocov = func(ancestry_poll, N=NK)
		for j in range(NK - np.sum(np.isnan(autocov))):
			across_running[k, j] += autocov[j]
			across_count[k, j] += 1
		k += 1

	# combine across-chromosome LAD with within-chromsomes LAD
	# across chromosome values stored in the last place
	running = np.append(running, across_running[:, 1:].sum(1).reshape(-1, 1), axis=1)
	count = np.append(count, across_count[:, 1:].sum(1).reshape(-1, 1), axis=1)

	return(running, count)


def get_ancestry_decay(ts, genetic_map, target_pop, func):
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
		interval_M=0.005,  # interval size in Morgans
		keep_interval=200,  # number of intervals to include
		func=func,
		all_other_pop=all_other_pop
	)
	return(running, count)
