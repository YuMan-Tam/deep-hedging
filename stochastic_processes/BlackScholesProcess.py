import QuantLib as ql
import numpy as np
from tqdm import trange

# References:
# https://www.implementingquantlib.com/2014/06/chapter-6-part-5-of-8-path-generators.html
# https://www.quantlib.org/reference/index.html
# https://github.com/lballabio/QuantLib-SWIG

# Assigned seed for testing. Set to 0 for random seeds.

# Geometric Brownian Motion.
class BlackScholesProcess:
	def __init__(self,s0 = None, sigma = None, risk_free = None, \
					dividend = None, day_count = None, seed=0):
                self.s0 = s0
                self.sigma = sigma
                self.risk_free = risk_free
                self.dividend = dividend
                self.day_count = day_count
                self.seed = seed
		
	def get_process(self, calculation_date = ql.Date.todaysDate()):
		spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.s0))
		rf_handle = ql.QuoteHandle(ql.SimpleQuote(self.risk_free))
		dividend_handle = ql.QuoteHandle(ql.SimpleQuote(self.dividend))
		
		volatility = ql.BlackConstantVol(0, ql.NullCalendar(),self.sigma,self.day_count)
		
		# Assume flat term-structure.
		flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), rf_handle, self.day_count))
		dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), dividend_handle, self.day_count))

		ql.Settings.instance().evaluationDate = calculation_date   

		return ql.GeneralizedBlackScholesProcess(
							spot_handle,
							dividend_yield,
							flat_ts,
							ql.BlackVolTermStructureHandle(volatility))

	def gen_path(self, length = None, time_step = None, num_paths = None):
		# The variable length is in the unit of one year.
		rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(time_step, ql.UniformRandomGenerator(self.seed)))
		seq = ql.GaussianMultiPathGenerator(self.get_process(), np.linspace(0,length,time_step+1), rng, False)
		
		value = np.zeros((num_paths, time_step+1))
		
		for i in trange(num_paths):
			sample_path = seq.next()
			path = sample_path.value()
			value[i, :] = np.array(path[0])
		return value                                  
