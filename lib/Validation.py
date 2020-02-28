import matplotlib as plt
import QuantLib as ql
import numpy as np
from Utilities import reshape_1D

class Validation:
	def __init__(self, model = None, data = None, N = None, process = None):
		self.model = model
		self.data = data
		self.process = process
		self.N = N
		
	def get_instrument(self, name = None, calculation_date = ql.Date.todaysDate(), **kwargs):
		if name is "European_Call":
			ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, kwargs["strike"])
			exercise_date = ql.EuropeanExercise(kwargs["maturity_date"])
			instrument = ql.VanillaOption(ql_payoff, exercise_date)

		if type(self.process).__name__ is "BlackScholesProcess":
			engine = ql.AnalyticEuropeanEngine(self.process.get_process(calculation_date))
			
		instrument.setPricingEngine(engine)
		return instrument
		
	def get_risk_neutral_PV(self, risk_free = None, dt = None, N = None, verbose = True):
		return -np.mean(self.data[-1])*np.exp(-risk_free*dt*N)
		
	def get_model_PV(self, instrument=None, s0=None, \
											calculation_date = ql.Date.todaysDate()):
		process = self.process
		process.s0 = s0
			
		engine = ql.AnalyticEuropeanEngine(process.get_process(calculation_date))
		instrument.setPricingEngine(engine)
		
		return instrument.NPV()

	def get_model_delta(self, instrument, s0=None, calculation_date = None):
		process = self.process
		process.s0 = s0
			
		engine = ql.AnalyticEuropeanEngine(process.get_process(calculation_date))
		instrument.setPricingEngine(engine)

		return instrument.delta()
