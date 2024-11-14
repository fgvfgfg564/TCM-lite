import json

def fix(filename, speedup):
    with open(filename, 'r') as f:
        d = json.load(f)
    
    s = f"speedup={speedup}"
    dnew = {k: {s: v} for k, v in d.items()}
    with open(filename, 'w') as f:
        json.dump(dnew, f)

fix("experiments/SAv1/performance/classa/results3-1.json", 2.0)
fix("experiments/SAv1/performance/classa/results3-2.json", 3.0)
fix("experiments/SAv1/performance/classa/results3-3.json", 4.0)