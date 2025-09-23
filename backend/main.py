from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional
import hashlib, io, base64, csv, math, random, time, os, json, datetime

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED or ["https://microchip-fabrication-tech.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # allow x-labtoken + content-type, etc.
)

@app.get("/health")
def health():
    return {"ok": True}

def check_token(provided: Optional[str]):
    if LABTOKEN and provided != LABTOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

ASSIGNMENT_ID = "ECSE322F25_HW4_OXIDATION"

# ---------- Models ----------
Ambient = Literal["dry", "wet"]
Orient  = Literal["100", "111"]

class Run(BaseModel):
    temp_C: int = Field(ge=900, le=1100)
    ambient: Ambient
    time_min: int = Field(ge=5, le=240)
    orientation: Orient
    preclean: bool

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
    csv_base64: str
    preview_plot_png_base64: str
    tech_note: str
    uid: str

# ---------- Simulator helpers ----------

def get_rng(student_id: str) -> random.Random:
    seed = int(hashlib.sha256((student_id + ASSIGNMENT_ID).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)

def deal_grove_thickness_nm(temp_C: int, ambient: str, time_min: int, x0_nm: float = 0.0) -> float:
    """
    Deal–Grove-like toy model. Returns final oxide thickness (nm) after time_min,
    starting from initial oxide x0_nm. Parameters are pedagogical and tunable.
    Solves x^2 + A x = (x0^2 + A x0) + B t  for x (units: A [um], B [um^2/min]).
    """
    A_um = {
        ('dry', 900): 0.080, ('dry', 1000): 0.070, ('dry', 1100): 0.060,
        ('wet', 900): 0.040, ('wet', 1000): 0.035, ('wet', 1100): 0.030,
    }
    B_um2_per_min = {
        ('dry', 900): 1.0e-4, ('dry', 1000): 2.0e-4, ('dry', 1100): 6.0e-4,
        ('wet', 900): 6.0e-4, ('wet', 1000): 2.0e-3, ('wet', 1100): 5.0e-3,
    }

    key = (ambient, temp_C)
    if key not in A_um or key not in B_um2_per_min:
        raise HTTPException(
            status_code=400,
            detail=f"temp_C must be one of [900, 1000, 1100] and ambient in [dry, wet]; got temp_C={temp_C}, ambient={ambient}."
        )

    A = A_um[key]
    B = B_um2_per_min[key]
    x0_um = max(0.0, float(x0_nm) / 1000.0)
    rhs = x0_um * x0_um + A * x0_um + B * float(time_min)
    disc = A * A + 4.0 * rhs
    x_um = (-A + math.sqrt(max(0.0, disc))) / 2.0
    return max(0.0, 1000.0 * x_um)  # nm

# ---------- Routes ----------
@app.post("/run_experiments", response_model=ExperimentResponse)
def run_experiments(req: ExperimentRequest, x_labtoken: Optional[str] = Header(None)):
    check_token(x_labtoken)

    # Optional initial oxide from target (e.g., pad oxide)
    x0_nm = 0.0
    if req.target and isinstance(req.target, dict) and "initial_oxide_nm" in req.target:
        try:
            x0_nm = float(req.target["initial_oxide_nm"])
        except Exception:
            raise HTTPException(status_code=400, detail="target.initial_oxide_nm must be numeric if provided.")

    rng = get_rng(req.student_id)
    rows = [("run_id","temp_C","ambient","time_min","orientation","preclean",
             "thick_center_nm","thick_edge_nm","nonuniformity_pct")]
    centers = []

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

    # CSV
    buf = io.StringIO(); csv.writer(buf).writerows(rows)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Plot
    plt.figure()
    plt.plot(range(1, len(centers)+1), centers, marker="o")
    plt.xlabel("Run"); plt.ylabel("Center thickness (nm)")
    plt.title("Oxide Growth — Center Thickness")
    plt.tight_layout()
    img = io.BytesIO(); plt.savefig(img, format="png"); plt.close()
    png_b64 = base64.b64encode(img.getvalue()).decode()

    # Log
    stamp = datetime.datetime.now().isoformat()
    line = f"{stamp},{req.student_id},{len(req.design.runs)},{json.dumps(req.target)}\n"
    print(f"[SUBMISSION] {line.strip()}")
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("timestamp,student_id,n_runs,target\n")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)

    note = (
        "Runs completed. Thickness increases with temperature/time; wet grows faster. "
        "Uniformity is typically worse for wet. Consider a confirmation near target and a uniformity check."
    )
    uid = f"{req.student_id}-{int(time.time())}"

    return ExperimentResponse(
        run_log=f"Processed {len(req.design.runs)} run(s).",
        csv_base64=csv_b64,
        preview_plot_png_base64=png_b64,
        tech_note=note,
        uid=uid
    )

@app.get("/admin/log")
def get_log(x_labtoken: Optional[str] = Header(None), token: Optional[str] = None):
    check_token(x_labtoken or token)
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("timestamp,student_id,n_runs,target\n", media_type="text/csv")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="text/csv")

@app.get("/admin/stats")
def stats(x_labtoken: Optional[str] = Header(None), token: Optional[str] = None):
    check_token(x_labtoken or token)
    counts = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                _, sid, _, _ = line.rstrip("\n").split(",", 4)
                counts[sid] = counts.get(sid, 0) + 1
    return JSONResponse({"by_student": counts, "total": sum(counts.values())})
