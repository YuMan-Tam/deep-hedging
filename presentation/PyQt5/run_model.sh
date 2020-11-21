# By Yu-Man Tam: 4/10/2020

# Command to run:
# bsub -Is -q "python"  numactl -C 0 zsh run_model.sh
# bsub -Is -q "python"  numactl --cpunodebind=0 zsh run_model.sh

# Run command:
# 	exec(open("deep_hedging_gui.py").read())

python3 
