Instruction for Running on the Grid:

Run:

bsub -Is -q "python"  numactl --cpunodebind=0 zsh python3

Then, within Python:

exec(open("deep_hedging_gui.py").read())
