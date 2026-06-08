"""
Ahmed body STL generator — headless FreeCAD script.

Usage (terminal):
  /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd generate_ahmed_freecad.py

Parameters are passed via environment variables (avoids FreeCAD arg parser conflicts):
  AHMED_SLANT_ANGLE     rear slant angle in degrees          (default: 25.0)
  AHMED_R_NOSE          leading-edge fillet radius in mm     (default: 100.0)
  AHMED_DIFFUSER_ANGLE  underbody diffuser angle in degrees  (default: 0.0)
  AHMED_L               body length in mm                    (default: 1044.0)
  AHMED_W               body width in mm                     (default: 389.0)
  AHMED_H               body height in mm                    (default: 288.0)
  AHMED_h               ground clearance in mm               (default: 50.8)
  AHMED_OUT             output STL path                      (default: see below)

Example DoE call:
  AHMED_SLANT_ANGLE=35 AHMED_R_NOSE=50 AHMED_DIFFUSER_ANGLE=12 AHMED_OUT=/tmp/case_01/ahmed.stl freecadcmd generate_ahmed_freecad.py
"""
import FreeCAD, Part, Mesh, MeshPart
import os, sys

# freecadcmd executes the script twice — guard against the second run
# if os.environ.get('_AHMED_RUNNING'):
#     sys.exit(0)
# os.environ['_AHMED_RUNNING'] = '1'

# ── Parameters from environment variables ────────────────────────────────────
def _env(key, default):
    return type(default)(os.environ.get(key, default))

slant_angle_deg     = _env('AHMED_SLANT_ANGLE',    25.0)
R_nose              = _env('AHMED_R_NOSE',         100.0)
diffuser_angle_deg  = _env('AHMED_DIFFUSER_ANGLE', 0.0)
L                   = _env('AHMED_L',              1044.0)
W                   = _env('AHMED_W',              389.0)
H                   = _env('AHMED_H',              288.0)
h                   = _env('AHMED_h',              50.8)

# ── Dimensions (all in mm for FreeCAD, exported as m) ────────────────────────
x_slant = 822.0    # x where rear slant begins
z_bot = h
z_top = h + H

# Feet (4 cylindrical legs, Lienhart 2002)
R_foot = 15.0
FEET = [
    (195.0,  97.25),
    (195.0, -97.25),
    (849.0,  97.25),
    (849.0, -97.25),
]

# Clamp R_nose: top and bottom arcs must not overlap (each needs R < H/2)
R_nose_max = H / 2.0 - 5.0
if R_nose > R_nose_max:
    print(f"WARNING: R_nose={R_nose}mm clamped to {R_nose_max:.1f}mm (H/2 - 5mm limit)")
    R_nose = R_nose_max

print(f"Generating Ahmed body: slant={slant_angle_deg}° R_nose={R_nose}mm diffuser={diffuser_angle_deg}° L={L} W={W} H={H} h={h}")

# ── Output path ───────────────────────────────────────────────────────────────
OUT = os.environ.get('AHMED_OUT') or os.path.expanduser(
    "~/Documents/aston_martin_cfd/cfmesh_validation/constant/triSurface/ahmed_body_hq.stl"
)

# ── 1. Main box ───────────────────────────────────────────────────────────────
box = Part.makeBox(L, W, H,
                   FreeCAD.Vector(0, -W/2, z_bot),
                   FreeCAD.Vector(0, 0, 1))

# ── 2. Slant cut — wedge that removes the rear top corner ────────────────────
import math
slant_rad = math.radians(slant_angle_deg)
slant_dx  = L - x_slant                     # 222 mm
slant_dz  = slant_dx * math.tan(slant_rad)  # ~103.5 mm

# Cutting prism: triangular cross-section in x-z plane, extruded in y.
# The three corners of the cut triangle:
#   A = (x_slant, z_top)          — top of slant start
#   B = (L,       z_top - slant_dz) — top of rear face (slant end)
#   C = (L,       z_top + 50)     — above body at rear (ensures full removal)
#   D = (x_slant, z_top + 50)     — above body at slant start
cut_pts = [
    FreeCAD.Vector(x_slant, 0, z_top),
    FreeCAD.Vector(L,       0, z_top - slant_dz),
    FreeCAD.Vector(L,       0, z_top + 50),
    FreeCAD.Vector(x_slant, 0, z_top + 50),
]
cut_wire = Part.makePolygon(cut_pts + [cut_pts[0]])
cut_face = Part.Face(cut_wire)
cut_solid = cut_face.extrude(FreeCAD.Vector(0, W + 100, 0))
cut_solid.translate(FreeCAD.Vector(0, -W/2 - 50, 0))

body = box.cut(cut_solid)

# ── 3. Diffuser cut — rear underbody ramp ────────────────────────────────────
# Start just after the rear feet (x=849+15+5=869 mm) to avoid conflicts.
x_rear_feet = max(fx for fx, fy in FEET)   # 849.0 mm
x_diffuser  = x_rear_feet + R_foot + 5.0   # ≈ 869 mm

if diffuser_angle_deg > 0.0:
    diff_rad = math.radians(diffuser_angle_deg)
    diffuser_dz = (L - x_diffuser) * math.tan(diff_rad)
    diff_pts = [
        FreeCAD.Vector(x_diffuser, 0, z_bot),
        FreeCAD.Vector(L,          0, z_bot),
        FreeCAD.Vector(L,          0, z_bot + diffuser_dz),
    ]
    diff_wire  = Part.makePolygon(diff_pts + [diff_pts[0]])
    diff_face  = Part.Face(diff_wire)
    diff_solid = diff_face.extrude(FreeCAD.Vector(0, W + 100, 0))
    diff_solid.translate(FreeCAD.Vector(0, -W/2 - 50, 0))
    body = body.cut(diff_solid)
    print(f"Diffuser cut: angle={diffuser_angle_deg}° x_start={x_diffuser:.0f}mm dz={diffuser_dz:.1f}mm")

# ── 5. Add feet ───────────────────────────────────────────────────────────────
# Cylinders from z=0 (ground) up to z=z_bot (body underside), fused to body
for (fx, fy) in FEET:
    foot = Part.makeCylinder(R_foot, h,
                             FreeCAD.Vector(fx, fy, 0),
                             FreeCAD.Vector(0, 0, 1))
    body = body.fuse(foot)

body = body.removeSplitter()  # clean up internal faces at fuse boundaries
print(f"Feet added at: {FEET}")

# ── 6. Fillet the leading edges ───────────────────────────────────────────────
# Find edges to fillet: the top-front edge and both side-front edges at x≈0
# FreeCAD filleting works on edge indices — we identify them by position.

fillet_edges = []
for i, edge in enumerate(body.Edges):
    # Each edge has a bounding box; leading edges are at x≈0
    bbox = edge.BoundBox
    mid = edge.CenterOfMass

    is_at_front = bbox.XMax < 1.0  # entirely at x=0

    if not is_at_front:
        continue

    # Top leading edge: horizontal at z≈z_top, spans full width
    is_top_edge = (abs(mid.z - z_top) < 1.0 and
                   abs(mid.y) < 1.0 and
                   bbox.YLength > W * 0.8)

    # Bottom leading edge: horizontal at z≈z_bot, spans full width
    is_bot_edge = (abs(mid.z - z_bot) < 1.0 and
                   abs(mid.y) < 1.0 and
                   bbox.YLength > W * 0.8)

    # Side leading edges: vertical at y≈±W/2, z spans body height
    is_side_edge = (abs(abs(mid.y) - W/2) < 1.0 and
                    bbox.ZLength > H * 0.5)

    if is_top_edge or is_bot_edge or is_side_edge:
        fillet_edges.append(i + 1)  # FreeCAD edge indices are 1-based

print(f"Edges selected for fillet: {fillet_edges}")

if len(fillet_edges) < 2:
    print("WARNING: could not find enough fillet edges — check geometry")
    print("Edges at x<1mm:")
    for i, e in enumerate(body.Edges):
        if e.BoundBox.XMax < 1.0:
            print(f"  Edge {i+1}: mid={e.CenterOfMass}, len={e.Length:.1f}")
else:
    body = body.makeFillet(R_nose, [body.Edges[i-1] for i in fillet_edges])
    print(f"Fillet applied at R={R_nose}mm on {len(fillet_edges)} edges")

# ── 7. Export STL ─────────────────────────────────────────────────────────────
# Scale from mm to m
scale = 1.0 / 1000.0
mat = FreeCAD.Matrix()
mat.scale(scale, scale, scale)
body_m = body.transformGeometry(mat)

mesh = MeshPart.meshFromShape(
    Shape=body_m,
    LinearDeflection=0.0005,   # 0.5 mm surface deviation
    AngularDeflection=0.1,     # ~6 deg angular deviation
    Relative=False
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
mesh.write(OUT)

ntri = mesh.CountFacets
print(f"\nDone. {ntri} triangles written to:")
print(f"  {OUT}")

# Quick bounds check
bb = body_m.BoundBox
print(f"\nBounds (metres):")
print(f"  X: {bb.XMin:.4f} → {bb.XMax:.4f}  (expect 0 → 1.044)")
print(f"  Y: {bb.YMin:.4f} → {bb.YMax:.4f}  (expect -0.1945 → 0.1945)")
print(f"  Z: {bb.ZMin:.4f} → {bb.ZMax:.4f}  (expect 0 → 0.3388, feet reach ground)")
