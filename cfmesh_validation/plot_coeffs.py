import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parallel runs write coefficient_0.dat (rank 0); serial runs write coefficient.dat.
# Pick whichever is larger (more data).
base = Path(__file__).parent / "postProcessing/forceCoeffs/0"
candidates = sorted(base.glob("coefficient*.dat"), key=lambda p: p.stat().st_size, reverse=True)
if not candidates:
    raise FileNotFoundError(f"No coefficient*.dat found in {base}")
coeff_file = candidates[0]
print(f"Reading: {coeff_file.name}  ({coeff_file.stat().st_size} bytes)")

data = np.loadtxt(coeff_file, comments="#")

iters = data[:, 0]
Cd    = data[:, 1]
Cl    = data[:, 4]

# Averaging window: last 500 iterations
window = min(100, len(iters))
Cd_mean = Cd[-window:].mean()
Cl_mean = Cl[-window:].mean()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

ax1.plot(iters, Cd, linewidth=0.8)
ax1.axhline(Cd_mean, color="red", linestyle="--", label=f"Mean (last {window}): {Cd_mean:.4f}")
ax1.set_ylabel("Cd")
ax1.legend()
ax1.grid(True)

ax2.plot(iters, Cl, linewidth=0.8)
ax2.axhline(Cl_mean, color="red", linestyle="--", label=f"Mean (last {window}): {Cl_mean:.4f}")
ax2.set_ylabel("Cl")
ax2.set_xlabel("Iteration")
ax2.legend()
ax2.grid(True)

plt.suptitle("Ahmed Body — Force Coefficients (kOmegaSST, 25° slant)")
plt.tight_layout()

out = Path(__file__).parent / "force_coeffs.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
print(f"Mean Cd (last {window} iters): {Cd_mean:.4f}  →  Cd_real ≈ {Cd_mean/2:.4f}")
print(f"Mean Cl (last {window} iters): {Cl_mean:.4f}  →  Cl_real ≈ {Cl_mean/2:.4f}")
plt.show()
