import sys, torch
sys.path.insert(0, '/challenge')
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
bench, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm09')
ct_pos = bench.macro_positions
ct_cost = compute_proxy_cost(ct_pos, bench, plc)['proxy_cost']
print(f"ibm09 CT: {ct_cost:.4f}")
