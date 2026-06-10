"""
generate_domain_stl.py
======================
Combines an Ahmed body STL (FreeCAD binary or ASCII output) with a
six-face ASCII domain bounding box into a single ASCII STL file suitable
for cfMesh's cartesianMesh utility.

Body triangles are split into two patches:
  ahmed_body      — all body surfaces except leg bases
  ahmed_legs_base — triangles whose centroid z < ride_height * 0.1
                    (the flat leg bottom faces that sit at ground level).
                    Excluded from BL insertion to prevent inverted cells
                    at the leg-ground junction.

The threshold scales with ride_height (read from AHMED_h env var, mm)
so it remains valid across the full DoE parameter space.

Domain boundary patches:
  inlet       — x = -5.22 m (upstream inflow)
  outlet      — x = +15.66 m (downstream outflow)
  ground      — z =  0.00 m (ground plane)
  top         — z =  2.88 m (top slip)
  side1       — y = -1.95 m (lateral slip)
  side2       — y = +1.95 m (lateral slip)

All normals follow the right-hand rule pointing OUTWARD from the
bounding surface (cfMesh convention for external flow).

Usage:
  python3 generate_domain_stl.py <body_stl_in> <combined_stl_out>

Domain bounds match the validated wind-tunnel analogue:
  upstream 5L, downstream 14L, lateral ±6W, top 9H — blockage ≈ 1%.
"""

import os
import struct
import sys
from pathlib import Path


# ── Domain bounds (m) ─────────────────────────────────────────────────────────
XMIN, XMAX = -5.22, 15.66
YMIN, YMAX = -1.95,  1.95
ZMIN, ZMAX =  0.00,  2.88


# ── STL I/O ───────────────────────────────────────────────────────────────────

def read_stl(path: Path) -> list[tuple]:
    """Return list of (nx, ny, nz, v1, v2, v3) for every triangle."""
    data = path.read_bytes()

    # Detect binary vs ASCII
    try:
        header = data[:80].decode("ascii", errors="ignore").strip()
    except Exception:
        header = ""

    if data[:5] == b"solid" and b"\nfacet" in data[:256]:
        return _read_ascii_stl(data.decode("ascii", errors="ignore"))
    else:
        return _read_binary_stl(data)


def _read_binary_stl(data: bytes) -> list[tuple]:
    n_tri = struct.unpack_from("<I", data, 80)[0]
    tris = []
    offset = 84
    for _ in range(n_tri):
        vals = struct.unpack_from("<12f", data, offset)
        nx, ny, nz = vals[0], vals[1], vals[2]
        v1 = (vals[3],  vals[4],  vals[5])
        v2 = (vals[6],  vals[7],  vals[8])
        v3 = (vals[9],  vals[10], vals[11])
        tris.append(((nx, ny, nz), v1, v2, v3))
        offset += 50
    return tris


def _read_ascii_stl(text: str) -> list[tuple]:
    tris = []
    lines = iter(text.splitlines())
    for line in lines:
        line = line.strip()
        if line.startswith("facet normal"):
            parts = line.split()
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            next(lines)  # outer loop
            verts = []
            for _ in range(3):
                vl = next(lines).strip().split()
                verts.append((float(vl[1]), float(vl[2]), float(vl[3])))
            tris.append((normal, verts[0], verts[1], verts[2]))
    return tris


def write_ascii_stl(f, name: str, triangles: list[tuple]):
    f.write(f"solid {name}\n")
    for (nx, ny, nz), v1, v2, v3 in triangles:
        f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
        f.write(f"    outer loop\n")
        f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
        f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
        f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
        f.write(f"    endloop\n")
        f.write(f"  endfacet\n")
    f.write(f"endsolid {name}\n")


# ── Body patch classification ─────────────────────────────────────────────────

def classify_body_triangles(tris: list[tuple], ride_height_m: float) -> dict[str, list[tuple]]:
    """Split body triangles into ahmed_body and ahmed_legs_base.

    Triangles whose centroid z < ride_height * 0.1 are leg base faces —
    they sit flush at ground level and must not receive BL extrusion.
    The threshold scales with ride_height so it holds across the DoE range.
    """
    threshold = ride_height_m * 0.1
    body, legs_base = [], []
    for tri in tris:
        _, v1, v2, v3 = tri
        centroid_z = (v1[2] + v2[2] + v3[2]) / 3.0
        if centroid_z < threshold:
            legs_base.append(tri)
        else:
            body.append(tri)
    return {"ahmed_body": body, "ahmed_legs_base": legs_base}


# ── Domain box face geometry ──────────────────────────────────────────────────
# Each rectangular face → 2 triangles.
# Vertices are ordered so the cross product of (v2-v1) × (v3-v1) gives the
# outward normal (pointing away from the interior fluid domain).

def _tri(n, a, b, c):
    return (n, a, b, c)

def domain_faces() -> dict[str, list[tuple]]:
    x0, x1 = XMIN, XMAX
    y0, y1 = YMIN, YMAX
    z0, z1 = ZMIN, ZMAX

    faces = {}

    # Inlet  x = x0,  outward normal = (-1, 0, 0)
    # CCW from -x: A(x0,y0,z0) → C(x0,y1,z1) → B(x0,y1,z0); A → D(x0,y0,z1) → C
    n = (-1.0, 0.0, 0.0)
    A = (x0, y0, z0); B = (x0, y1, z0); C = (x0, y1, z1); D = (x0, y0, z1)
    faces["inlet"] = [_tri(n, A, C, B), _tri(n, A, D, C)]

    # Outlet  x = x1,  outward normal = (+1, 0, 0)
    # CCW from +x
    n = (1.0, 0.0, 0.0)
    A = (x1, y0, z0); B = (x1, y1, z0); C = (x1, y1, z1); D = (x1, y0, z1)
    faces["outlet"] = [_tri(n, A, B, C), _tri(n, A, C, D)]

    # Ground  z = z0,  outward normal = (0, 0, -1)
    n = (0.0, 0.0, -1.0)
    A = (x0, y0, z0); B = (x1, y0, z0); C = (x1, y1, z0); D = (x0, y1, z0)
    faces["ground"] = [_tri(n, A, C, B), _tri(n, A, D, C)]

    # Top  z = z1,  outward normal = (0, 0, +1)
    n = (0.0, 0.0, 1.0)
    A = (x0, y0, z1); B = (x1, y0, z1); C = (x1, y1, z1); D = (x0, y1, z1)
    faces["top"] = [_tri(n, A, B, C), _tri(n, A, C, D)]

    # Side1  y = y0,  outward normal = (0, -1, 0)
    n = (0.0, -1.0, 0.0)
    A = (x0, y0, z0); B = (x1, y0, z0); C = (x1, y0, z1); D = (x0, y0, z1)
    faces["side1"] = [_tri(n, A, B, C), _tri(n, A, C, D)]

    # Side2  y = y1,  outward normal = (0, +1, 0)
    n = (0.0, 1.0, 0.0)
    A = (x0, y1, z0); B = (x1, y1, z0); C = (x1, y1, z1); D = (x0, y1, z1)
    faces["side2"] = [_tri(n, A, C, B), _tri(n, A, D, C)]

    return faces


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <body_stl_in> <combined_stl_out>")
        sys.exit(1)

    body_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    ride_height_mm = float(os.environ.get("AHMED_h", "50.8"))
    ride_height_m  = ride_height_mm / 1000.0
    leg_threshold  = ride_height_m * 0.1
    print(f"  ride_height={ride_height_mm} mm  →  leg-base threshold z < {leg_threshold*1000:.2f} mm")

    print(f"Reading body STL: {body_path}")
    body_tris = read_stl(body_path)
    print(f"  {len(body_tris)} triangles")

    patches = classify_body_triangles(body_tris, ride_height_m)
    for name, tris in patches.items():
        print(f"  {name}: {len(tris)} triangles")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    domain = domain_faces()
    print(f"Writing combined STL: {out_path}")
    with open(out_path, "w") as f:
        for patch_name, tris in patches.items():
            if tris:
                write_ascii_stl(f, patch_name, tris)
        for patch_name, tris in domain.items():
            write_ascii_stl(f, patch_name, tris)

    total = len(body_tris) + sum(len(t) for t in domain.values())
    print(f"  {total} triangles total")
    print(f"  Domain patches: {', '.join(domain.keys())}")
    print(f"  Domain: x=[{XMIN}, {XMAX}]  y=[{YMIN}, {YMAX}]  z=[{ZMIN}, {ZMAX}]")


if __name__ == "__main__":
    main()
