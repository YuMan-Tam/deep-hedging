import QuantLib as ql
import numpy as np

# References:
# https://www.implementingquantlib.com/2014/06/chapter-6-part-5-of-8-path-generators.html
# https://www.quantlib.org/reference/index.html
# https://github.com/lballabio/QuantLib-SWIG

# Assigned seed for testing. Set to 0 for random seeds.
seed = 0

# Geometric Brownian Motion.
class BlackScholesProcess:
	def __init__(self,s0 = None, sigma = None, risk_free = None, dividend = None):
		self.s0 = s0
		self.sigma = sigma
		self.risk_free = risk_free
		self.dividend = dividend

	def get_process(self, calculation_date = ql.Date.todaysDate()):
		day_count = ql.Actual365Fixed()
		spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.s0))
		rf_handle = ql.QuoteHandle(ql.SimpleQuote(self.risk_free))
		dividend_handle = ql.QuoteHandle(ql.SimpleQuote(self.dividend))
		
		volatility = ql.BlackConstantVol(0, ql.NullCalendar(),self.sigma,day_count)
		
		# Assume flat term-structure.
		flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), rf_handle, day_count))
		dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), dividend_handle, day_count))

		ql.Settings.instance().evaluationDate = calculation_date   

		return ql.GeneralizedBlackScholesProcess(
							spot_handle,
							dividend_yield,
							flat_ts,
							ql.BlackVolTermStructureHandle(volatility))

	def gen_path(self, length = None, time_step = None, num_paths = None):
		# The variable length is in the unit of one year.
		rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(time_step, ql.UniformRandomGenerator(seed)))
		seq = ql.GaussianMultiPathGenerator(self.get_process(), np.linspace(0,length,time_step+1), rng, False)
		
		value = np.zeros((num_paths, time_step+1))
		
		for i in range(num_paths):
			sample_path = seq.next()
			path = sample_path.value()
			value[i, :] = np.array(path[0])
		return value                                  

# Heston process
class HestonProcess:
    def __init__(self, s0 = None, v0 = None, \
                    kappa = None, theta = None, sigma = None, rho = None, \
                    risk_free = None, dividend = None):
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.risk_free = risk_free
        self.dividend = dividend
        
    def get_process(self, calculation_date = ql.Date.todaysDate()):
        day_count = ql.Actual365Fixed()
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.s0))
        
        # Assume flat dividend yield curve and flat term-structure.
        # We need to be very careful of the calender treatment. ql.TARGET() does not work here!
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), self.risk_free, day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), self.dividend, day_count))
        
        ql.Settings.instance().evaluationDate = calculation_date # DD/MM/YYYY
        
        # The variable length is in the unit of one year.
        return ql.HestonProcess(flat_ts,
                          dividend_yield,
                          spot_handle,
                          self.v0,
                          self.kappa,
                          self.theta,
                          self.sigma,
                          self.rho)
        
    def gen_path(self, length = None, time_step = None, num_paths = None):            
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator((time_step)*2, ql.UniformRandomGenerator(seed)))
        
        # See https://github.com/lballabio/quantlib-old/blob/master/QuantLib-SWIG/SWIG/montecarlo.i
        # Note that dividing an interval by n would contains n+1 points.
        seq = ql.GaussianMultiPathGenerator(self.get_process(), np.linspace(0,length,time_step+1), rng, False)
        
        S = np.zeros((num_paths, time_step+1))
        V = np.zeros((num_paths, time_step+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()

            S[i, :] = np.array(path[0])
            V[i, :] = np.array(path[1])
        
        return S, V
        
    def get_VarSwap_path(self, V = None, length = None):
        [num_paths, time_step] = V.shape 
        # See P.21 of Buehler et al (2019)
        realized_variance = np.cumsum(V, axis=1)*(length/(time_step-1))
        
        time2maturity = length - np.linspace(0,length,time_step)
        time2maturity = np.tile(time2maturity,(num_paths,1))
        L = np.multiply(((V - self.theta)/self.kappa),(1-np.exp(-self.kappa*time2maturity))) + \
                self.theta*time2maturity
        return realized_variance + L
    
