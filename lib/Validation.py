import matplotlib as plt
import QuantLib as ql
import numpy as np
from Utilities import reshape_1D

class Validation:
	def __init__(self, model = None, data = None, process = None):
		self.model = model
		self.data = data
		self.process = process
		self.N = int((len(data) - 2 - 1)/2) # Subtract two dimension for w and payoff, 
									   # and the last day for the information set.
		
	def get_instrument(self, name = None, calculation_date = ql.Date.todaysDate(), **kwargs):
		if name is "European_Call":
			ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, kwargs["strike"])
			exercise_date = ql.EuropeanExercise(kwargs["maturity_date"])
			instrument = ql.VanillaOption(ql_payoff, exercise_date)

		if type(self.process).__name__ is "HestonProcess":
			engine = ql.AnalyticHestonEngine(ql.HestonModel(self.process.get_process(calculation_date)))
		elif type(self.process).__name__ is "BlackScholesProcess":
			engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
			
		instrument.setPricingEngine(engine)
		return instrument
		
	def get_risk_neutral_PV(self, verbose = True):
		risk_neutral_price = -np.mean(self.data[-1])
		if verbose:
			print("The risk neutral price for the European call is {0}.".format(risk_neutral_price))
		return risk_neutral_price
		
	def get_model_PV(self, instrument=None, verbose = True):
		model_NPV = instrument.NPV()
		if verbose:
			print("The model price for the European call is {0}.".format(model_NPV))
		return model_NPV

	def get_model_delta(self, instrument,\
					 s0 = None, v0 = None, calculation_date = None, \
					 Warning = False, min_v = 1e-8,  h = 1e-12):
		process = self.process
		process.s0 = s0
			
		if type(self.process).__name__ is "HestonProcess":
			# To prevent pricing failure at v = 0.0.
			# Check value of std::numeric_limits<double>::epsilon()
			process.v0 = max(v0,min_v)
			delta = 0.0
					
			for sign in [1.0, -1.0]:
				process.s0 = s0 + sign*h
				engine = ql.AnalyticHestonEngine(ql.HestonModel(process.get_process(calculation_date)))
				instrument.setPricingEngine(engine)
				
				delta += instrument.NPV()*sign
				
			return delta/(2.0*h)
		elif type(self.process).__name__ is "BlackScholesProcess":
			engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
			instrument.setPricingEngine(engine)
			
			return instrument.delta()
		
	def get_model_vega(self, instrument, \
					 s0 = None, v0 = None, calculation_date = None, \
					 Warning = False, min_v = 1e-8, h = 1e-12):
		process = self.process
		
		process.s0 = s0
		vega = 0.0
		
		for sign in [1.0, -1.0]:
			# To prevent pricing failure at v = 0.0.
			process.v0 = max(v0,min_v) + sign*h
			engine = ql.AnalyticHestonEngine(ql.HestonModel(process.get_process(calculation_date)))
			instrument.setPricingEngine(engine)
			
			# For variance swap, see equation (5.4) of Buehler (2019)
			VarSwap_prc = instrument.NPV() 
			
			vega += VarSwap_prc*sign
			
		partial_v_u = vega/(2.0*h)
		
		is_VarSwap = True

		if is_VarSwap:
			time2maturity = (ql.Date.todaysDate() + self.N - calculation_date)/365.0
			partial_v_L = (1.0/process.kappa)*(1 - np.exp(-process.kappa*time2maturity))
		
			return partial_v_u / partial_v_L
		else:
			return partial_v_u
		
	def get_model_wealth(self, initial_wealth = None, model_delta = None, model_vega = None):
		model_wealth = np.ones(len(self.data[0]))*initial_wealth
		contingent_claim_prc = -self.data[-1]
	
		if type(self.process).__name__ is "BlackScholesProcess":
			model_strategy = model_delta[:,0]
			for j in range(self.N+1):
				hedge_asset_prc = self.data[2*j+1]
				
				if j == 0:
					model_wealth += np.multiply(hedge_asset_prc,model_strategy)
				elif j < self.N:
					model_strategy_lag1 = model_strategy
					model_strategy = model_delta[:,j]
					model_wealth += np.multiply(hedge_asset_prc,model_strategy-model_strategy_lag1)
				else:
					model_wealth += np.multiply(hedge_asset_prc,-model_strategy)
					model_wealth += contingent_claim_prc
					
		elif type(self.process).__name__ is "HestonProcess":
			model_strategy = np.transpose(np.vstack([model_delta[:,0], model_vega[:,0]]))
					
			for j in range(self.N+1):
				hedge_asset_prc = self.data[2*j+1]
				
				if j == 0:
					model_wealth += np.sum(np.multiply(hedge_asset_prc,model_strategy), axis=1)
				elif j < self.N:
					model_strategy_lag1 = model_strategy
					model_strategy = np.transpose(np.vstack([model_delta[:,j], model_vega[:,j]]))
					model_wealth += np.sum(np.multiply(hedge_asset_prc,model_strategy-model_strategy_lag1), axis=1)
				else:
					model_wealth += np.sum(np.multiply(hedge_asset_prc,-model_strategy), axis=1)
					model_wealth += contingent_claim_prc
	  
		return model_wealth
