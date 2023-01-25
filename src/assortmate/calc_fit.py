import numpy as np
import scipy as sp
import numba
import warnings
import argparse


def cli():
	parser = argparse.ArgumentParser(description="Calculate fit")

	parser.add_argument(
		'-p', '--basepath',
		type=str, required=True,
		help='basepath to input files'
	)
	parser.add_argument(
		'-n', '--nboot',
		type=int, default=1000, required=False,
		help='number of bootstrap replicates'
	)

	return parser.parse_args()


def Haldane(M):
	"""Convert from Morgans to recombination fraction with the Haldane mapping function.

	M -- numpy array of positions given in Morgans
	"""
	c = 0.5 * (1 - np.exp(-2 * M))
	return c


def main():
	args = cli()
	basepath = args.basepath
	nboot = args.nboot

	tern = np.load(f'{basepath}.tern.npz')['tern']
	running = np.load(f'{basepath}.lad.npz')['running']
	count = np.load(f'{basepath}.lad.npz')['count']
	decay = running.sum(0) / count.sum(0)

	# should be the same as passed to calc_LAD
	cM_interval = 0.5
	cM_max = 100
	M = np.arange(0, cM_max / 100, cM_interval / 100)
	H = Haldane(M)
	# nkeep = len(np.arange(0, cM_max / 100, cM_interval / 100))
	Hc = np.concatenate([H, [0.5]])

	# remove the cross-chromosome term
	running = running[:, :-1]
	count = count[:, :-1]
	decay = decay[:-1]
	Hc = Hc[:-1]

	@numba.njit
	def Xfunc(c, intercept, R, G):
		"""
		Function to fit ancestry decay.

		The variables alpha, beta, V, and LAD0 should be defined outside the function.

		c -- recombination fraction
		intercept -- the intercept
		R -- correlation in ancestry between mates
		G -- number of generations since admixture
		"""
		rho0 = R * LAD0  # intial covariance in ancestry between mates
		phase = (2 * V * R) / (1 + R)  # additional LAD due to phase switching
		a = (1 - c)**G * LAD0  # LAD due to no recombination
		b = c * rho0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))  # LAD with recombination
		AM = (
			intercept +
			a +
			b +
			phase +   # addition matching due to unphasing
			alpha**2 +  # random matching
			beta**2  # random matching
		)
		return AM

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

	def xnext_fk(G, R):
		# intial values
		G = G + 1.0  # adjust the generations to match the two methods
		f = 1.0
		k = 1.0
		# this weighting provides a linear interpolation between generations
		Gi = int(np.floor(G))
		weight = G - Gi
		for i in range(Gi):
			f, k, R = next_fk(f, k, R)
		# one more for interpolation
		fi, ki, R = next_fk(f, k, R)
		f = f * (1 - weight) + fi * weight
		return f

	def twostep():
		step1_popt, step1_pcov = sp.optimize.curve_fit(
			f=Xfunc,  # function relating input to observed data F(x, params) -> y
			xdata=Hc,  # genetic distances (in recombination fraction) where decay is observed
			ydata=decay,  # observed data
			p0=[0, .1, 10],  # initial parameter values
			bounds=([-1, 0, 1], [1, 0.99, np.inf]),  # bounds on the parameters
			sigma=(running.std(0) / np.sqrt(count.sum(0))),  # estimated errors in decay
			absolute_sigma=False,  # is sigma in units of decay
		)
		step1_perr = np.sqrt(np.diag(step1_pcov))
		intercept, R_est1, G_est1 = step1_popt

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)
			step2_popt, step2_pcov = sp.optimize.curve_fit(
				xnext_fk,
				xdata=np.array([G_est1]),
				ydata=np.array([f]),
				p0=R_est1,
				bounds=([0], [0.99]),
			)
			R_est2 = step2_popt[0]

		step2_perr = np.sqrt(np.diag(step2_pcov))
		return(intercept, R_est1, R_est2, G_est1)

	def fit(running, count, tern, maxiter=100, epsilon=0.001):
		alpha = tern.mean(0)[0] + tern.mean(0)[1] / 2  # initial admixture proportion
		Q = tern[:, 0] + tern[:, 1] / 2
		V = np.var(Q)
		beta = 1 - alpha
		LAD0 = alpha * beta
		EXP_HET = 2 * alpha * beta
		OBS_HET = tern[:, 1].mean()
		f = 1 - OBS_HET / EXP_HET
		flag = False
		if f <= 0:
			# if f is negative, set it to a small positive value
			f = 0.001
			flag = True

		@numba.njit
		def Xfunc(c, intercept, R, G):
			"""
			Function to fit ancestry decay.

			The variables alpha, beta, and LAD0 should be defined outside the function.

			c -- recombination fraction
			intercept -- the intercept
			R -- correlation in ancestry between mates
			G -- number of generations since admixture
			"""
			rho0 = R * LAD0  # intial covariance in ancestry between mates
			phase = (2 * V * R) / (1 + R)  # additional LAD due to phase switching
			a = (1 - c)**G * LAD0  # LAD due to no recombination
			b = c * rho0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))  # LAD with recombination
			AM = (
				intercept +
				a +
				b +
				phase +   # addition matching due to unphasing
				alpha**2 +  # random matching
				beta**2  # random matching
			)
			return AM

		@numba.njit
		def Wfunc(c, intercept, G, R):
			"""
			Function to fit ancestry decay.

			The variables alpha, beta, and LAD0 should be defined outside the function.

			c -- recombination fraction
			intercept -- the intercept
			R -- correlation in ancestry between mates
			G -- number of generations since admixture
			"""
			rho0 = R * LAD0  # intial covariance in ancestry between mates
			phase = (2 * V * R) / (1 + R)  # additional LAD due to phase switching
			a = (1 - c)**G * LAD0  # LAD due to no recombination
			b = c * rho0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))  # LAD with recombination
			AM = (
				intercept +
				a +
				b +
				phase +   # addition matching due to unphasing
				alpha**2 +  # random matching
				beta**2  # random matching
			)
			return AM

		def twostep():
			step1_popt, step1_pcov = sp.optimize.curve_fit(
				f=Xfunc,  # function relating input to observed data F(x, params) -> y
				xdata=Hc,  # genetic distances (in recombination fraction) where decay is observed
				ydata=decay,  # observed data
				p0=[0, .1, 10],  # initial parameter values
				bounds=([-1, 0, 1], [1, 0.99, np.inf]),  # bounds on the parameters
				sigma=(running.std(0) / np.sqrt(count.sum(0))),  # estimated errors in decay
				absolute_sigma=False,  # sigma is in units of decay
			)
			# step1_perr = np.sqrt(np.diag(step1_pcov))
			intercept, R_est1, G_est1 = step1_popt

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)
				step2_popt, step2_pcov = sp.optimize.curve_fit(
					xnext_fk,
					xdata=np.array([G_est1]),
					ydata=np.array([f]),
					p0=R_est1,
					bounds=([0], [0.99]),
				)
				R_est2 = step2_popt[0]

			# step2_perr = np.sqrt(np.diag(step2_pcov))
			return(intercept, R_est1, R_est2, G_est1)

		EEfunc = lambda c, intercept, G: Wfunc(c, intercept, G, R=R_est2)

		delta = 1
		vals = np.zeros((maxiter, 4))

		# initial fit
		intercept1, R_est1, R_est2, G_est1 = twostep()

		vals[0] = [0, R_est2, G_est1, intercept1]

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)

			for i in range(1, maxiter):
				step1_popt, step1_pcov = sp.optimize.curve_fit(
					f=EEfunc,  # function relating input to observed data F(x, params) -> y
					xdata=Hc,  # genetic distances (in recombination fraction) where decay is observed
					ydata=decay,  # observed data
					p0=[intercept1, G_est1],  # initial parameter values
					bounds=([-1, 1], [1, np.inf]),  # bounds on the parameters
					sigma=(running.std(0) / np.sqrt(count.sum(0))),  # estimated errors in decay
					absolute_sigma=False,  # sigma is in units of decay
				)

				intercept1, G_est1 = step1_popt
				step1_perr = np.sqrt(np.diag(step1_pcov))

				step2_popt, step2_pcov = sp.optimize.curve_fit(
					f=xnext_fk,
					xdata=np.array([G_est1]),
					ydata=np.array([f]),
					p0=R_est2,
					bounds=([0], [0.99]),
				)
				R_est2 = step2_popt[0]

				step2_perr = np.sqrt(np.diag(step2_pcov))
				vals[i] = [i, R_est2, G_est1, intercept1]
				delta = np.mean(np.abs(vals[i, 1:3] - vals[i - 1, 1:3]))
				if (delta < epsilon) and (i > 9):
					break

		R_est2, G_est1, intercept1 = vals[i, 1:4]

		return np.array([alpha, f, OBS_HET, intercept1, R_est2, G_est1, flag, i])

	def resamplefit(running, count, tern, maxiter=100, epsilon=0.001):

		alpha = tern.mean(0)[0] + tern.mean(0)[1] / 2  # initial admixture proportion
		Q = tern[:, 0] + tern[:, 1] / 2
		V = np.var(Q)
		beta = 1 - alpha
		# initial values
		LAD0 = alpha * beta
		EXP_HET = 2 * alpha * beta
		OBS_HET = tern[:, 1].mean()
		f = 1 - OBS_HET / EXP_HET
		flag = False
		if f <= 0:
			# if f is negative, set it to a small positive value
			f = 0.001
			flag = True

		@numba.njit
		def Xfunc(c, intercept, R, G):
			"""
			Function to fit ancestry decay.

			The variables alpha, beta, and LAD0 should be defined outside the function.

			c -- recombination fraction
			intercept -- the intercept
			R -- correlation in ancestry between mates
			G -- number of generations since admixture
			"""
			rho0 = R * LAD0  # intial covariance in ancestry between mates
			phase = (2 * V * R) / (1 + R)  # additional LAD due to phase switching
			a = (1 - c)**G * LAD0  # LAD due to no recombination
			b = c * rho0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))  # LAD with recombination
			AM = (
				intercept +
				a +
				b +
				phase +   # addition matching due to unphasing
				alpha**2 +  # random matching
				beta**2  # random matching
			)
			return AM

		@numba.njit
		def Wfunc(c, intercept, G, R):
			"""
			Function to fit ancestry decay.

			The variables alpha, beta, and LAD0 should be defined outside the function.

			c -- recombination fraction
			intercept -- the intercept
			R -- correlation in ancestry between mates
			G -- number of generations since admixture
			"""
			rho0 = R * LAD0  # intial covariance in ancestry between mates
			phase = (2 * V * R) / (1 + R)  # additional LAD due to phase switching
			a = (1 - c)**G * LAD0  # LAD due to no recombination
			b = c * rho0 * ((1 + R)**G - (1 - c)**G * 2**G) / (2**(G - 1) * (R + 2 * c - 1))  # LAD with recombination
			AM = (
				intercept +
				a +
				b +
				phase +   # addition matching due to unphasing
				alpha**2 +  # random matching
				beta**2  # random matching
			)
			return AM

		def twostep():
			step1_popt, step1_pcov = sp.optimize.curve_fit(
				f=Xfunc,  # function relating input to observed data F(x, params) -> y
				xdata=Hc,  # genetic distances (in recombination fraction) where decay is observed
				ydata=decay,  # observed data
				p0=[0, .1, 10],  # initial parameter values
				bounds=([-1, 0, 1], [1, 0.99, np.inf]),  # bounds on the parameters
				sigma=(running.std(0) / np.sqrt(count.sum(0))),  # estimated errors in decay
				absolute_sigma=False,  # is sigma in units of decay
			)
			step1_perr = np.sqrt(np.diag(step1_pcov))
			intercept, R_est1, G_est1 = step1_popt

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)
				step2_popt, step2_pcov = sp.optimize.curve_fit(
					xnext_fk,
					xdata=np.array([G_est1]),
					ydata=np.array([f]),
					p0=R_est1,
					bounds=([0], [0.99]),
				)
				R_est2 = step2_popt[0]

			step2_perr = np.sqrt(np.diag(step2_pcov))
			return(intercept, R_est1, R_est2, G_est1)

		EEfunc = lambda c, intercept, G: Wfunc(c, intercept, G, R=R_est2)

		delta = 1
		vals = np.zeros((maxiter, 4))

		# initial fit
		intercept1, R_est1, R_est2, G_est1 = twostep()

		vals[0] = [0, R_est2, G_est1, intercept1]

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=sp.optimize.OptimizeWarning)

			for i in range(1, maxiter):
				step1_popt, step1_pcov = sp.optimize.curve_fit(
					f=EEfunc,  # function relating input to observed data F(x, params) -> y
					xdata=Hc,  # genetic distances (in recombination fraction) where decay is observed
					ydata=decay,  # observed data
					p0=[intercept1, G_est1],  # initial parameter values
					bounds=([-1, 1], [1, np.inf]),  # bounds on the parameters
					sigma=(running.std(0) / np.sqrt(count.sum(0))),  # estimated errors in decay
					absolute_sigma=False,  # sigma is in units of decay
				)

				intercept1, G_est1 = step1_popt
				step1_perr = np.sqrt(np.diag(step1_pcov))

				step2_popt, step2_pcov = sp.optimize.curve_fit(
					f=xnext_fk,
					xdata=np.array([G_est1]),
					ydata=np.array([f]),
					p0=R_est2,
					bounds=([0], [0.99]),
				)
				R_est2 = step2_popt[0]

				step2_perr = np.sqrt(np.diag(step2_pcov))
				vals[i] = [i, R_est2, G_est1, intercept1]
				delta = np.mean(np.abs(vals[i, 1:3] - vals[i - 1, 1:3]))
				if (delta < epsilon) and (i > 9):
					break

		R_est2, G_est1, intercept1 = vals[i, 1:4]

		return np.array([alpha, f, OBS_HET, intercept1, R_est2, G_est1])

	def myboot(data, nboot):
		running, count, tern = data
		theta_boot = np.zeros((nboot, 6))
		nd = tern.shape[0]
		for i in range(nboot):
			b = np.random.choice(nd, size=nd, replace=True)
			theta_boot[i] = resamplefit(running[b], count[b], tern[b])
		return(theta_boot)

	def myjack(data):
		running, count, tern = data
		nd = tern.shape[0]
		theta_jack = np.zeros((nd, 6))

		for i in range(nd):
			theta_jack[i] = resamplefit(
				np.delete(running, i, axis=0),
				np.delete(count, i, axis=0),
				np.delete(tern, i, axis=0),
			)
		return(theta_jack)

	def get_alpha(ci_width):
		alpha = (1 - ci_width) / 2
		return alpha

	def get_ahat(jn):
		jns = jn.mean(axis=0)
		num = np.sum((jns - jn)**3, axis=0)
		denom = np.sum((jns - jn)**2, axis=0)
		ahat = num / (6 * (denom**(3 / 2)))
		return(ahat)

	def get_zhat(theta, theta_boot):
		zhat = sp.stats.norm.ppf(np.mean(theta_boot < theta, axis=0))
		return(zhat)

	def bca(ahat, zhat, ci_width):
		alpha = get_alpha(ci_width)

		za = sp.stats.norm.ppf(alpha)
		low = zhat + (zhat + za) / (1 - ahat * (zhat + za))

		za = sp.stats.norm.ppf(1 - alpha)
		high = zhat + (zhat + za) / (1 - ahat * (zhat + za))

		return np.array([sp.stats.norm.cdf(low), sp.stats.norm.cdf(high)])

	def bca_bootstrap(data, nboot, ci_width=0.95):
		running, count, tern = data
		theta = resamplefit(running, count, tern)

		theta_boot = myboot(data, nboot=nboot)
		theta_jack = myjack(data)

		ahat = get_ahat(theta_jack)
		zhat = get_zhat(theta, theta_boot)
		lowhigh = bca(ahat, zhat, ci_width)

		n = 6
		ci = np.zeros((2, n))
		for j in range(n):
			try:
				pp = np.quantile(
					a=theta_boot[:, j],
					q=lowhigh[:, j]).round(3)
			except ValueError:
				pp = -9
			ci[:, j] = pp
		return ci, theta, theta_boot, theta_jack, ahat, zhat, lowhigh

	# # what to save
	# theta
	# ci
	# theta_boot
	# theta_jack
	# ahat
	# zhat
	# lowhigh

	data = [running, count, tern]

	theta = fit(running, count, tern).round(3)

	if nboot > 0:
		res = bca_bootstrap(data, nboot=nboot)
		# unpack
		ci, resampletheta, theta_boot, theta_jack, ahat, zhat, lowhigh = res
	else:
		ci = np.zeros(1)
		theta_boot = np.zeros(1)
		theta_jack = np.zeros(1),
		ahat = np.zeros(1)
		zhat = np.zeros(1)
		lowhigh = np.zeros(1)

	np.savez_compressed(
		file=f'{basepath}.fit.npz',
		theta=theta,
		ci=ci,
		theta_boot=theta_boot,
		theta_jack=theta_jack,
		ahat=ahat,
		zhat=zhat,
		lowhigh=lowhigh,
	)


if __name__ == '__main__':
	main()
