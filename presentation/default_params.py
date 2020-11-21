# Define the initial parameters for the deep hedging demo
def DeepHedgingParams():
	params = [
			{'name': 'European Call Option', 'type': 'group', 'children': [
					{'name': 'S0', 'type': 'int', 'value': 100.0},
					{'name': 'Strike', 'type': 'float', 'value': 100.0},
					{'name': 'Implied Volatility', 'type': 'float', 'value': 0.2},
					{'name': 'Risk-Free Rate', 'type': 'float', 'value': 0.0},
					{'name': 'Dividend Yield', 'type': 'float', 'value': 0.0},
					{'name': 'Maturity (in days)', 'type': 'int', 'value': 30},
					{'name': 'Proportional Transaction Cost', 'type': 'group', 'children': [
            {'name': 'Cost', 'type': 'float', 'value': 0.0},
        ]},
			]},
			{'name': 'Monte-Carlo Simulation', 'type': 'group', 'children': [
					{'name': 'Sample Size', 'type': 'group', 'children': [
            {'name': 'Training', 'type': 'int', 'value': 1*(10**5)},
            {'name': 'Testing (as fraction of Training)', 'type': 'float', 'value': 0.2}
					]},
			]},
			{'name': 'Deep Hedging Strategy', 'type': 'group', 'children': [
					{'name': 'Loss Function', 'type': 'group', 'children': [
            {'name': 'Loss Type', 'type': 'list', 'values': {"Entropy": "Entropy", "CVaR": "CVaR"}, "default": "Entropy"},
            {'name': 'Risk Aversion', 'type': 'float', 'value': 1.0}
					]},
					{'name': 'Network Structure', 'type': 'group', 'children': [
						{'name': 'Network Type', 'type': 'list', 'values': {"Simple": "simple", "Recurrent": "recurrent"}, "default": "simple"},
            {'name': 'Number of Hidden Layers', 'type': 'int', 'value': 1},
            {'name': 'Number of Neurons', 'type': 'int', 'value': 15},
					]},
					{'name': 'Learning Parameters', 'type': 'group', 'children': [
            {'name': 'Learning Rate', 'type': 'float', 'value': 5e-3},
            {'name': 'Mini-Batch Size', 'type': 'int', 'value': 256},
            {'name': 'Number of Epochs', 'type': 'int', 'value': 50},
					]},
			]},
	]
	return params
