from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Dict, Tuple
import hashlib, io, base64, csv, math, random, time, os, json, datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ---------- App & settings ----------
app = FastAPI(title="Oxidation Technician")

ALLOWED  = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LABTOKEN = os.getenv("LABTOKEN", "")

# Logs: persist to /data if you add a Render Disk, else current dir
LOG_DIR  = "/data" if os.path.isdir("/data") else "."
LOG_PATH = os.path.join(LOG_DIR, "submissions_log.csv")

# Rate limit (per student_id)
RL_WINDOW_S = int(os.getenv("RL_WINDOW_S", "300"))  # 5 min default
RL_MAX_HITS = int(os.getenv("RL_MAX_HITS", "10"))   # 10 requests per window
_BUCKET: Dict[str, List[float]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED or ["https://microchip-fabrication-tech.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # allow x-labtoken + content-type, etc.
)

FRONTEND_ORIGIN = (ALLOWED[0] if ALLOWED else "https://microchip-fabrication-tech.onrender.com").strip()

@app.options("/run_experiments")
def options_run_experiments():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": FRONTEND_ORIGIN,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, x-labtoken",
            "Access-Control-Max-Age": "600",
        },
    )

@app.get("/health")
def health():
    return {"ok": True}

def check_token(provided: Optional[str]):
    if LABTOKEN and provided != LABTOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

def check_rate_limit(student_id: str):
    now = time.time()
    sid = (student_id or "").strip().lower()
    if not sid:
        raise HTTPException(status_code=400, detail="student_id required")
    hits = _BUCKET.get(sid, [])
    hits = [t for t in hits if (now - t) < RL_WINDOW_S]  # drop old
    if len(hits) >= RL_MAX_HITS:
        raise HTTPException(status_code=429, detail="Too many submissions; try again later.")
    hits.append(now)
    _BUCKET[sid] = hits

ASSIGNMENT_ID = "ECSE322F25_HW4_OXIDATION"

# ---------- Models ----------
Ambient = Literal["dry", "wet"]
Orient  = Literal["100", "111"]

class Run(BaseModel):
    temp_C: int = Field(ge=800, le=1200)
    ambient: Ambient
    time_min: int = Field(ge=5, le=240)
    orientation: Orient
    preclean: bool
    @validator("temp_C")
    def temp_step(cls, v):
        if v % 10 != 0:
            raise ValueError("temp_C must be a multiple of 10 between 800 and 1200")
        return v

class Design(BaseModel):
    strategy: Literal["screening", "confirmation", "optimization"]
    runs: List[Run]
    @validator("runs")
    def run_count(cls, v):
        if not (1 <= len(v) <= 12):
            raise ValueError("runs must have 1–12 items")
        return v

class ExperimentRequest(BaseModel):
    objective: str
    target: Optional[dict] = None  # may include {"thickness_nm":..., "tolerance_nm":..., "initial_oxide_nm":...}
    design: Design
    notes: Optional[str] = None
    student_id: str
    section: str

class ExperimentResponse(BaseModel):
    run_log: str
    csv_base64: str                     # per-run summary (center, edge, % nonuniformity)
    map_csv_base64: str                 # 3×3 map per run
    preview_plot_png_base64: str        # center thickness vs run
    wafer_maps_png_base64: str          # composite wafer heatmaps (all runs)
    tech_note: str
    uid: str

# ---------- Deal–Grove via Arrhenius ----------
KB_eV_per_K = 8.617333262e-5  # Boltzmann (eV/K)

# Distinct pedagogical reference values (not from the slide)
# Units: B_ref [um^2/min], BA_ref [um/min], Tref_C [°C], Ea [eV]
DG_REF = {
    "dry": {"Tref_C": 1000, "B_ref": 3.2e-4, "BA_ref": 0.014, "Ea_B_eV": 1.15, "Ea_BA_eV": 2.05},
    "wet": {"Tref_C": 1000, "B_ref": 6.2, "BA_ref": 1620000, "Ea_B_eV": 0.75, "Ea_BA_eV": 1.45},
}

def _arrhenius(value_ref: float, Ea_eV: float, T_C: float, Tref_C: float) -> float:
    T = 273.15 + float(T_C)
    Tref = 273.15 + float(Tref_C)
    return value_ref * math.exp(-Ea_eV/KB_eV_per_K * (1.0/T))

def dg_params_from_arrhenius(ambient: str, temp_C: int) -> Tuple[float, float, float]:
    """Return (A_um, B_um2_per_min, BA_um_per_min) using Arrhenius scaling."""
    if ambient not in DG_REF:
        raise HTTPException(400, detail=f"ambient must be 'dry' or 'wet'; got {ambient}")
    ref = DG_REF[ambient]
    B  = _arrhenius(ref["B_ref"],  ref["Ea_B_eV"],  temp_C, ref["Tref_C"])
    BA = _arrhenius(ref["BA_ref"], ref["Ea_BA_eV"], temp_C, ref["Tref_C"])
    BA = max(1e-9, BA)  # guard
    A  = B / BA
    return A, B, BA

def deal_grove_thickness_nm(temp_C: int, ambient: str, time_min: int, x0_nm: float = 0.0) -> float:
    """
    Solve x^2 + A x = (x0^2 + A x0) + B t   for x (um), then return nm.
    """
    A_um, B_um2_per_min, _ = dg_params_from_arrhenius(ambient, temp_C)
    x0_um = max(0.0, float(x0_nm)/1000.0)
    rhs = x0_um*x0_um + A_um*x0_um + B_um2_per_min*float(time_min)
    disc = A_um*A_um + 4.0*rhs
    x_um = (-A_um + math.sqrt(max(0.0, disc))) / 2.0
    return max(0.0, 1000.0*x_um)

def get_rng(student_id: str) -> random.Random:
    seed = int(hashlib.sha256((student_id + ASSIGNMENT_ID).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)

# ---------- Route ----------
@app.post("/run_experiments", response_model=ExperimentResponse)
def run_experiments(req: ExperimentRequest, x_labtoken: Optional[str] = Header(None)):
    check_token(x_labtoken)
    check_rate_limit(req.student_id)

    # Optional initial oxide (pad oxide)
    x0_nm = 0.0
    if req.target and isinstance(req.target, dict) and "initial_oxide_nm" in req.target:
        try:
            x0_nm = float(req.target["initial_oxide_nm"])
        except Exception:
            raise HTTPException(status_code=400, detail="target.initial_oxide_nm must be numeric if provided.")

    rng = get_rng(req.student_id)

    rows = [("run_id","temp_C","ambient","time_min","orientation","preclean",
             "thick_center_nm","thick_edge_nm","nonuniformity_pct")]
    map_rows = [("run_id","p00","p01","p02","p10","p11","p12","p20","p21","p22")]
    map_arrays: List[np.ndarray] = []
    centers: List[float] = []

    for i, r in enumerate(req.design.runs, start=1):
        # Deal–Grove base (nm)
        base = deal_grove_thickness_nm(r.temp_C, r.ambient, r.time_min, x0_nm=x0_nm)

        # Systematic effects
        orient_k = 0.90 if r.orientation == "111" else 1.00
        pre_k    = 1.05 if r.preclean else 1.00
        mean = base * orient_k * pre_k

        # Noise (~4% with occasional extra wiggle)
        base_noise_sigma = 0.04
        big_bump = 1.0 + abs(rng.gauss(0, 0.5)) * 0.01
        noise = rng.gauss(0, base_noise_sigma) * big_bump
        center = max(0.0, mean * (1 + noise))

        # Nonuniformity: edge thinner; wet generally worse
        nonuni_base = 0.02 + (0.03 if r.ambient == "wet" else 0.015)
        nonuni = nonuni_base * (1 + abs(rng.gauss(0, 0.6)))
        edge = max(0.0, center * (1 - nonuni))

        rows.append((i, r.temp_C, r.ambient, r.time_min, r.orientation, r.preclean,
                     round(center,1), round(edge,1), round(100*nonuni,1)))
        centers.append(center)

        # 3×3 map (radial thinning scaled by nonuniformity)
        grid = np.zeros((3,3), dtype=float)
        maxr = math.hypot(1,1)
        for a in range(3):
            for b in range(3):
                rnorm = math.hypot(a-1, b-1) / maxr  # 0..1
                shape = 0.6 + 0.4 * rnorm
                grid[a, b] = max(0.0, center * (1 - nonuni * shape))
        map_arrays.append(grid)
        map_rows.append((i, *[round(grid[a,b],1) for a in range(3) for b in range(3)]))

    # Summary CSV
    buf = io.StringIO(newline="")
    csv.writer(buf).writerows(rows)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Map CSV
    mbuf = io.StringIO(newline="")
    csv.writer(mbuf).writerows(map_rows)
    map_csv_b64 = base64.b64encode(mbuf.getvalue().encode()).decode()

    # Composite wafer-map PNG (heatmaps per run)
    wafer_maps_png_base64 = ""
    if map_arrays:
        n = len(map_arrays)
        cols = min(4, n)
        rows_fig = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_fig, cols, figsize=(3.2*cols, 3.2*rows_fig))
        axes = np.atleast_2d(axes)
        vmin = min(a.min() for a in map_arrays)
        vmax = max(a.max() for a in map_arrays)
        im = None
        for idx, arr in enumerate(map_arrays):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"Run {idx+1}")
            ax.set_xticks([]); ax.set_yticks([])
        # hide unused subplots
        for j in range(n, rows_fig*cols):
            axes[j//cols, j%cols].axis("off")
        fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)
        if im is not None:
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cax, label="Thickness (nm)")
        buf_img = io.BytesIO()
        fig.savefig(buf_img, format="png", dpi=150)
        plt.close(fig)
        wafer_maps_png_base64 = base64.b64encode(buf_img.getvalue()).decode()


    # Plot (center thickness vs run)
    plt.figure()
    plt.plot(range(1, len(centers)+1), centers, marker="o")
    plt.xlabel("Run"); plt.ylabel("Center thickness (nm)")
    plt.title("Oxide Growth — Center Thickness")
    plt.tight_layout()
    img = io.BytesIO(); plt.savefig(img, format="png"); plt.close()
    png_b64 = base64.b64encode(img.getvalue()).decode()


    # ---- Logging (file + Render logs) ----
    stamp = datetime.datetime.now().isoformat()
    record = {
        "timestamp": stamp,
        "student_id": req.student_id,
        "n_runs": len(req.design.runs),
        "target": req.target
    }
    print(f"[SUBMISSION] {json.dumps(record, separators=(',',':'))}")

    header = ["timestamp","student_id","n_runs","target_json"]
    need_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow([stamp, req.student_id, len(req.design.runs), json.dumps(req.target)])

    note = (
        "Runs completed. Thickness increases with temperature and time; wet grows faster. "
        "Edge regions tend to be thinner; wet usually shows higher nonuniformity. "
        "Consider a confirmation near target and a uniformity check."
    )
    uid = f"{req.student_id}-{int(time.time())}"

    return ExperimentResponse(
        run_log=f"Processed {len(req.design.runs)} run(s).",
        csv_base64=csv_b64,
        map_csv_base64=map_csv_b64,
        preview_plot_png_base64=png_b64,
        wafer_maps_png_base64=wafer_maps_png_base64,
        tech_note=note,
        uid=uid
    )

# ---------- Admin (view logs) ----------
@app.get("/admin/log")
def get_log(x_labtoken: Optional[str] = Header(None), token: Optional[str] = None):
    check_token(x_labtoken or token)
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("timestamp,student_id,n_runs,target_json\n", media_type="text/csv")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="text/csv")

@app.get("/admin/stats")
def stats(x_labtoken: Optional[str] = Header(None), token: Optional[str] = None):
    check_token(x_labtoken or token)
    counts: Dict[str, int] = {}
    total = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if not row: continue
                sid = row[1]
                counts[sid] = counts.get(sid, 0) + 1
                total += 1
    return JSONResponse({"by_student": counts, "total": total})




