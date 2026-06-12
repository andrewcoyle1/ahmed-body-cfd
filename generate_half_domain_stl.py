"""
generate_half_domain_stl.py
===========================
Generates a half-domain STL (y = 0 → +1.95 m) for cfMesh symmetry cases.

Key difference from the naive centroid-filter approach:
- Triangles are properly CLIPPED at y=0 (not just filtered by centroid)
- The open body edge at y=0 is CAPPED with a flat polygon → watertight half-body
- cfMesh therefore sees a closed surface and applies localRefinement correctly

Usage:
  python3 generate_half_domain_stl.py <body_stl_in> <combined_stl_out>
"""

import math
import struct
import sys
from pathlib import Path

XMIN, XMAX = -5.22, 15.66
YMIN, YMAX =  0.00,  1.95
ZMIN, ZMAX =  0.00,  2.88
RIDE_HEIGHT_M = 0.0508


# ── STL I/O ───────────────────────────────────────────────────────────────────

def read_stl(path):
    data = path.read_bytes()
    try:
        if "facet normal" in data[:1024].decode("ascii"):
            return _read_ascii(data.decode("ascii", errors="ignore"))
    except Exception:
        pass
    return _read_binary(data)

def _read_binary(data):
    n = struct.unpack_from("<I", data, 80)[0]
    tris, offset = [], 84
    for _ in range(n):
        if offset + 48 > len(data):
            break
        v = struct.unpack_from("<12f", data, offset)
        tris.append(((v[0],v[1],v[2]), (v[3],v[4],v[5]), (v[6],v[7],v[8]), (v[9],v[10],v[11])))
        offset += 50
    return tris

def _read_ascii(text):
    tris, lines = [], iter(text.splitlines())
    for line in lines:
        if line.strip().startswith("facet normal"):
            p = line.split()
            n = (float(p[2]), float(p[3]), float(p[4]))
            next(lines)
            verts = [tuple(float(x) for x in next(lines).split()[1:4]) for _ in range(3)]
            tris.append((n, *verts))
    return tris

def write_ascii_stl(f, name, tris):
    f.write(f"solid {name}\n")
    for (nx,ny,nz), v1, v2, v3 in tris:
        f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n"
                f"    outer loop\n"
                f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
                f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
                f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n"
                f"    endloop\n"
                f"  endfacet\n")
    f.write(f"endsolid {name}\n")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def norm(v):
    m = math.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return (v[0]/m, v[1]/m, v[2]/m) if m > 1e-30 else v

def tri_normal(v1, v2, v3):
    return norm(cross(sub(v2,v1), sub(v3,v1)))

def interp_y0(a, b):
    """Intersection of segment a→b with the y=0 plane."""
    t = a[1] / (a[1] - b[1])
    return (a[0]+t*(b[0]-a[0]), 0.0, a[2]+t*(b[2]-a[2]))


# ── Triangle clipping at y=0 ─────────────────────────────────────────────────

def clip_triangle(tri):
    """Clip one triangle at y=0. Returns list of 0–2 triangles (y>=0 side)
    and a list of cut edges (pairs of points on y=0 plane)."""
    _, v1, v2, v3 = tri
    verts = [v1, v2, v3]
    above = [v[1] >= -1e-10 for v in verts]
    n_above = sum(above)

    cut_edges = []

    if n_above == 3:
        return [tri], []
    if n_above == 0:
        return [], []

    if n_above == 1:
        pi = above.index(True)
        p  = verts[pi]
        q1 = verts[(pi+1) % 3]
        q2 = verts[(pi+2) % 3]
        i1 = interp_y0(p, q1)
        i2 = interp_y0(p, q2)
        n  = tri_normal(p, i1, i2)
        cut_edges.append((i1, i2))
        return [(n, p, i1, i2)], cut_edges

    if n_above == 2:
        ni = above.index(False)
        q  = verts[ni]
        p1 = verts[(ni+1) % 3]
        p2 = verts[(ni+2) % 3]
        i1 = interp_y0(p1, q)
        i2 = interp_y0(p2, q)
        n1 = tri_normal(p1, p2, i1)
        n2 = tri_normal(p2, i2, i1)
        cut_edges.append((i1, i2))
        return [(n1,p1,p2,i1), (n2,p2,i2,i1)], cut_edges

    return [], []


# ── Cap polygon assembly ──────────────────────────────────────────────────────

def build_cap(cut_edges):
    """Given a list of (a, b) edges on y=0, assemble closed loops and
    triangulate them as a fan from the centroid. Normal points in -y
    direction (outward from the fluid domain).

    Edges are treated as UNDIRECTED — winding is determined per-loop
    from the desired -y outward normal."""
    if not cut_edges:
        return []

    def rnd(v):
        return (round(v[0], 7), round(v[2], 7))

    # Build undirected adjacency: vertex_key → list of (neighbour_key, neighbour_pt, edge_idx)
    from collections import defaultdict
    adj = defaultdict(list)
    for i, (a, b) in enumerate(cut_edges):
        ka, kb = rnd(a), rnd(b)
        adj[ka].append((kb, b, i))
        adj[kb].append((ka, a, i))

    used_edges = [False] * len(cut_edges)
    loops = []

    for start_key in list(adj.keys()):
        # Find an unused edge from this vertex
        start_candidates = [(nb_k, nb_pt, ei) for nb_k, nb_pt, ei in adj[start_key]
                            if not used_edges[ei]]
        if not start_candidates:
            continue

        nb_k, nb_pt, ei = start_candidates[0]
        used_edges[ei] = True

        # Recover actual start point
        a0, b0 = cut_edges[ei]
        if rnd(a0) == start_key:
            start_pt, loop = a0, [a0]
            current_key, current_pt = nb_k, nb_pt
        else:
            start_pt, loop = b0, [b0]
            current_key, current_pt = nb_k, nb_pt

        for _ in range(len(cut_edges)):
            candidates = [(nb_k2, nb_pt2, ei2) for nb_k2, nb_pt2, ei2 in adj[current_key]
                          if not used_edges[ei2]]
            if not candidates:
                break
            nb_k2, nb_pt2, ei2 = candidates[0]
            used_edges[ei2] = True
            loop.append(current_pt)
            current_key, current_pt = nb_k2, nb_pt2
            if current_key == rnd(start_pt):
                break

        if len(loop) >= 3:
            loops.append(loop)

    print(f"  Cap: {len(loops)} loops, {sum(len(l) for l in loops)} vertices")

    # Fan triangulation from centroid; fix winding so normal points in -y
    cap_tris = []
    for loop in loops:
        cx = sum(v[0] for v in loop) / len(loop)
        cz = sum(v[2] for v in loop) / len(loop)
        centroid = (cx, 0.0, cz)
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i+1) % len(loop)]
            n = tri_normal(centroid, a, b)
            if n[1] > 0:
                a, b = b, a
                n = tri_normal(centroid, a, b)
            cap_tris.append((n, centroid, a, b))

    return cap_tris


# ── Domain box ────────────────────────────────────────────────────────────────

def _tri(n, a, b, c):
    return (n, a, b, c)

def domain_faces():
    x0, x1 = XMIN, XMAX
    y0, y1 = YMIN, YMAX
    z0, z1 = ZMIN, ZMAX
    faces = {}

    n=(-1.,0.,0.); A=(x0,y0,z0); B=(x0,y1,z0); C=(x0,y1,z1); D=(x0,y0,z1)
    faces["inlet"] = [_tri(n,A,C,B), _tri(n,A,D,C)]

    n=(1.,0.,0.); A=(x1,y0,z0); B=(x1,y1,z0); C=(x1,y1,z1); D=(x1,y0,z1)
    faces["outlet"] = [_tri(n,A,B,C), _tri(n,A,C,D)]

    n=(0.,0.,-1.); A=(x0,y0,z0); B=(x1,y0,z0); C=(x1,y1,z0); D=(x0,y1,z0)
    faces["ground"] = [_tri(n,A,C,B), _tri(n,A,D,C)]

    n=(0.,0.,1.); A=(x0,y0,z1); B=(x1,y0,z1); C=(x1,y1,z1); D=(x0,y1,z1)
    faces["top"] = [_tri(n,A,B,C), _tri(n,A,C,D)]

    # symmetry at y=0, outward normal = (0,-1,0)
    n=(0.,-1.,0.); A=(x0,y0,z0); B=(x1,y0,z0); C=(x1,y0,z1); D=(x0,y0,z1)
    faces["symmetry"] = [_tri(n,A,B,C), _tri(n,A,C,D)]

    # side2 at y=y1, outward normal = (0,+1,0)
    n=(0.,1.,0.); A=(x0,y1,z0); B=(x1,y1,z0); C=(x1,y1,z1); D=(x0,y1,z1)
    faces["side2"] = [_tri(n,A,C,B), _tri(n,A,D,C)]

    return faces


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <body_stl_in> <combined_stl_out>")
        sys.exit(1)

    body_path = Path(sys.argv[1])
    out_path  = Path(sys.argv[2])

    print(f"Reading: {body_path}")
    all_tris = read_stl(body_path)
    print(f"  {len(all_tris)} triangles")

    threshold_z = RIDE_HEIGHT_M * 0.1
    body_tris, legs_base_tris = [], []
    all_cut_edges = []

    for tri in all_tris:
        clipped, cut_edges = clip_triangle(tri)
        all_cut_edges.extend(cut_edges)
        for ct in clipped:
            _, v1, v2, v3 = ct
            cz = (v1[2]+v2[2]+v3[2]) / 3.0
            if cz < threshold_z:
                legs_base_tris.append(ct)
            else:
                body_tris.append(ct)

    cap_tris = build_cap(all_cut_edges)
    print(f"  ahmed_body: {len(body_tris)} tris  legs_base: {len(legs_base_tris)} tris  cap: {len(cap_tris)} tris")

    domain = domain_faces()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing: {out_path}")
    with open(out_path, "w") as f:
        write_ascii_stl(f, "ahmed_body",      body_tris)
        write_ascii_stl(f, "ahmed_legs_base", legs_base_tris)
        for name, tris in domain.items():
            write_ascii_stl(f, name, tris)

    print("Done.")

if __name__ == "__main__":
    main()
