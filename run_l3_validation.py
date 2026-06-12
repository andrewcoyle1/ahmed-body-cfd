"""
Single L3 validation run at the confirmed optimum design:
  slant=32.5°, diffuser=8.9°, ride_height=50.8mm, front_radius=100mm
"""
import importlib.util, sys
from pathlib import Path

BASE = Path(__file__).parent

# Load runner module
spec = importlib.util.spec_from_file_location(
    "cfmesh_doe_runner", BASE / "cfmesh_doe_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# Switch to L3 mesh template
runner.L2_TEMPLATE = BASE / "mesh_convergence" / "L3_fine"

OPTIMUM = {
    "slant_angle":    32.492,
    "diffuser_angle":  8.948,
    "ride_height":    50.8,
    "front_radius":  100.0,
}

CASES_DIR = BASE / "openfoam_cases_l3_validation"
CASES_DIR.mkdir(exist_ok=True)

res = runner.run_case("case_l3_opt", OPTIMUM, resume=False, cases_dir=CASES_DIR)
print(f"\n{'='*50}")
print(f"L3 validation result:")
print(f"  Cd = {res['Cd']}")
print(f"  Cl = {res['Cl']}")
if res['Cd'] is not None and res['Cl'] is not None:
    f = res['Cd'] + (1/3) * res['Cl']
    print(f"  f  = {f:.6f}")
print(f"{'='*50}")
