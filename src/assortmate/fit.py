"""Functions for fitting ancestry decay and ancestry heterozygosity."""
import numba
import numpy as np

LAD0 = None
alpha = None
beta = None
R_est2 = None


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

	See Fig. 1 in Crow and Felesenstein (1968).

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
	"""Uses the estimate of R from step 2.

	c -- recombination fraction
	intercept -- the intercept
	G -- number of generations since admixture
	"""
	R = R_est2
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


@numba.njit
def Haldane(M):
	"""Convert from Morgans to recombination fraction with the Haldane mapping function.

	M -- numpy array of positions given in Morgans
	"""
	c = 0.5 * (1 - np.exp(-2 * M))
	return c


@numba.njit
def inv_Haldane(c):
	"""Convert from recombination fraction to Morgans with the inverse Haldane mapping function.

	c -- numpy array of positions given in recombination fractions.
	"""
	M = np.abs(np.log(1 - 2 * c) / - 2)
	return M


@numba.njit
def generate_query_points_c(max_c, interval):
	"""Generate an array of points (in Morgans) to query for LAD

	max_c -- maximum recombination fraction between sites to query.
	interval -- interval between adjacent query points (in units of recombination fraction)
	"""
	assert max_c < 0.5
	query_M = inv_Haldane(np.arange(0, max_c + interval, interval))
	return query_M


@numba.njit
def generate_query_points_M(max_M, interval):
	"""Generate an array of points (in Morgans) to query for LAD

	max_M -- maximum recombination fraction between sites to query.
	interval -- interval between adjacent query points (in units of Morgans)
	"""
	query_M = np.arange(0, max_M + interval, interval)
	return query_M
