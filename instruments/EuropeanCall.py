import QuantLib as ql
import numpy as np
from scipy import stats
from stochastic_processes import BlackScholesProcess

# Assume continuous dividend with flat term-structure and flat dividend structure.
class EuropeanCall:
	def __init__(self):
		pass
		
	def get_BS_price(self,S=None, sigma = None,risk_free = None, \
												dividend = None, K = None, exercise_date = None, calculation_date = None, \
												day_count = None, dt = None, evaluation_method = "Numpy"):
		
		if evaluation_method is "QuantLib":
			# For our purpose, assume all inputs are scalar.
			stochastic_process = BlackScholesProcess(s0 = S, sigma = sigma, \
												risk_free = risk_free, dividend = dividend, day_count=day_count)
				
			engine = ql.AnalyticEuropeanEngine(stochastic_process.get_process(calculation_date))
			
			ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
			exercise_date = ql.EuropeanExercise(exercise_date)
			instrument = ql.VanillaOption(ql_payoff, exercise_date)

			if type(self.process).__name__ is "BlackScholesProcess":
				engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
				
			instrument.setPricingEngine(engine)
			
			return instrument.NPV()
		elif evaluation_method is "Numpy":
			# For our purpose, assume s0 is a NumPy array, other inputs are scalar.
			T = np.arange(0, (exercise_date - calculation_date + 1))*dt
			T = np.repeat(np.flip(T[None,:]), S.shape[0], 0)
			
			# Ignore division by 0 warning (expected behaviors as the limits of CDF is defined).
			with np.errstate(divide='ignore'):
				d1 = np.divide(np.log(S / K) + (risk_free - dividend + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
				d2 = np.divide(np.log(S / K) + (risk_free - dividend - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			
			return (S * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-risk_free * T) * stats.norm.cdf(d2, 0.0, 1.0))
					
	def get_BS_delta(self,S=None, sigma = None,risk_free = None, \
												dividend = None, K = None, exercise_date = None, calculation_date = None, \
												day_count = None, dt = None, evaluation_method = "Numpy"):
		
		if evaluation_method is "QuantLib":
			# For our purpose, assume all inputs are scalar.
			stochastic_process = BlackScholesProcess(s0 = S, sigma = sigma, \
												risk_free = risk_free, dividend = dividend, day_count=day_count)
				
			engine = ql.AnalyticEuropeanEngine(stochastic_process.get_process(calculation_date))
			
			ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
			exercise_date = ql.EuropeanExercise(exercise_date)
			instrument = ql.VanillaOption(ql_payoff, exercise_date)

			if type(self.process).__name__ is "BlackScholesProcess":
				engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
				
			instrument.setPricingEngine(engine)
			
			return instrument.delta()
		elif evaluation_method is "Numpy":
			# For our purpose, assume s0 is a NumPy array, other inputs are scalar.
			T = np.arange(0, (exercise_date - calculation_date + 1))*dt
			T = np.repeat(np.flip(T[None,:]), S.shape[0], 0)
			
			# Ignore division by 0 warning (expected behaviors as the limits of CDF is defined).
			with np.errstate(divide='ignore'):
				d1 = np.divide(np.log(S / K) + (risk_free - dividend + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			
			return stats.norm.cdf(d1, 0.0, 1.0)
			
	def get_BS_vega(self,S=None, sigma = None,risk_free = None, \
												dividend = None, K = None, exercise_date = None, calculation_date = None, \
												day_count = None, dt = None, evaluation_method = "Numpy"):
		
		if evaluation_method is "QuantLib":
			# For our purpose, assume all inputs are scalar.
			stochastic_process = BlackScholesProcess(s0 = S, sigma = sigma, \
												risk_free = risk_free, dividend = dividend, day_count=day_count)
				
			engine = ql.AnalyticEuropeanEngine(stochastic_process.get_process(calculation_date))
			
			ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
			exercise_date = ql.EuropeanExercise(exercise_date)
			instrument = ql.VanillaOption(ql_payoff, exercise_date)

			if type(self.process).__name__ is "BlackScholesProcess":
				engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
				
			instrument.setPricingEngine(engine)
			
			return instrument.vega()
		elif evaluation_method is "Numpy":
			# For our purpose, assume s0 is a NumPy array, other inputs are scalar.
			T = np.arange(0, (exercise_date - calculation_date + 1))*dt
			T = np.repeat(np.flip(T[None,:]), S.shape[0], 0)
			
			# Ignore division by 0 warning (expected behaviors as the limits of CDF is defined).
			with np.errstate(divide='ignore'):
				d1 = np.divide(np.log(S / K) + (risk_free - dividend + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T))
			
			return np.multiply(S, np.sqrt(T))*stats.norm.pdf(d1, 0.0, 1.0)
			
	def get_BS_PnL(self, S = None, payoff = None, delta = None, dt = None, risk_free = None, \
									final_period_cost = None, epsilon = None, cost_structure="proportional"):
		# Compute Black-Scholes PnL (for a short position, i.e. the Bank sells
		# a call option. The model delta from Quantlib is a long delta.
		N = S.shape[1]-1
		
		PnL_BS = np.multiply(S[:,0], -delta[:,0]) \
		
		if cost_structure == "proportional":
			PnL_BS -= np.abs(delta[:,0])*S[:,0]*epsilon
		elif cost_structure == "constant":
			PnL_BS -= epsilon
				
		PnL_BS = PnL_BS*np.exp(risk_free*dt)
		
		for t in range(1, N):
			PnL_BS += np.multiply(S[:,t], -delta[:,t] + delta[:,t-1])
			
			if cost_structure == "proportional":
				PnL_BS -= np.abs(delta[:,t] -delta[:,t-1])*S[:,t]*epsilon
			elif cost_structure == "constant":
				PnL_BS -= epsilon
				
			PnL_BS = PnL_BS*np.exp(risk_free*dt)

		PnL_BS += np.multiply(S[:,N],delta[:,N-1]) + payoff 
		
		if final_period_cost:
			if cost_structure == "proportional":
				PnL_BS -= np.abs(delta[:,N-1])*S[:,N]*epsilon
			elif cost_structure == "constant":
				PnL_BS -= epsilon
				
		return PnL_BS
