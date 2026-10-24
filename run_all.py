"""Run the reconstructed reproducibility workflow."""
import subprocess,sys
from pathlib import Path
base=Path(__file__).resolve().parent
for name in ('generate_data.py','train_evaluate.py','analysis_outputs.py','stress_test.py'):
    print('\nRUNNING',name,flush=True)
    subprocess.run([sys.executable,str(base/name)],cwd=base,check=True)
print('\nCompleted reconstructed experiments. Compare outputs to manuscript before submission.')
