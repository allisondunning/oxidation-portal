from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional
import hashlib, io, base64, csv, math, random, time, os, json, datetime

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ---- FastAPI app ----
app = FastAPI(title="Oxidation Technician")

# Settings from environment (Render → Environment)
ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LABTOKEN = os.getenv("LABTOKEN", "")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED or ["*"],      # tighten to your exact site in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "x-labtoken"],
)

@app.get("/health")
def health():
    return {"ok": True}

ASSIGNMENT_ID = "ECSE322F25_HW4_OXIDATION"

# ---- Models ----
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
    target: Optional[dict] = None
    design: Design
    notes: Optional[str] = None
    student_id: str
    section: str

class ExperimentResponse(BaseModel):
    run_log: str
    csv_base64: str
    preview_plot_png_base64: str
    tech_note: str
    uid: str  # unique result id

# ---- Core simulator ----
def get_rng(student_id: str) -> random.Random:
    seed = int(hashlib.sha256((student_id + ASSIGNMENT_ID).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)

def growth_nm(temp_C: int, ambient: str, time_min: int) -> float:
    # Plausible rate table (nm/min). Adjust anytime.
    DRY = {900:0.6, 1000:1.0, 1100:3.3}
    WET = {900:2.0, 1000:5.0, 1100:10.0}
    base = (DRY if ambient == "dry" else WET)[temp_C]
    return base * time_min

@app.post("/run_experiments", response_model=ExperimentResponse)
def run_experiments(req: ExperimentRequest, x_labtoken: Optional[str] = Header(None)):
    # Token gate (optional but recommended)
    if LABTOKEN and x_labtoken != LABTOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

    rng = get_rng(req.student_id)

    rows = [("run_id","temp_C","ambient","time_min","orientation","preclean",
             "thick_center_nm","thick_edge_nm","nonuniformity_pct")]
    centers = []

    for i, r in enumerate(req.design.runs, start=1):
        mean = growth_nm(r.temp_C, r.ambient, r.time_min)

        # Systematic effects
        orient_k = 0.90 if r.orientation == "111" else 1.00
        pre_k    = 1.05 if r.preclean else 1.00

        # Noise (~4% with occasional extra wiggle)
        base_noise_sigma = 0.04
        big_bump = 1.0 + abs(rng.gauss(0, 0.5)) * 0.01
        noise = rng.gauss(0, base_noise_sigma) * big_bump

        center = max(0.0, mean * orient_k * pre_k * (1 + noise))

        # Edge thinner; worse for wet; add variability
        nonuni_base = 0.02 + (0.03 if r.ambient == "wet" else 0.015)
        nonuni = nonuni_base * (1 + abs(rng.gauss(0, 0.6)))
        edge = max(0.0, center * (1 - nonuni))

        rows.append((i, r.temp_C, r.ambient, r.time_min, r.orientation, r.preclean,
                     round(center,1), round(edge,1), round(100*nonuni,1)))
        centers.append(center)

    # CSV
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    csv_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    # Plot
    plt.figure()
    plt.plot(range(1, len(centers)+1), centers, marker="o")
    plt.xlabel("Run"); plt.ylabel("Center thickness (nm)")
    plt.title("Oxide Growth — Center Thickness")
    plt.tight_layout()
    img = io.BytesIO(); plt.savefig(img, format="png"); plt.close()
    png_b64 = base64.b64encode(img.getvalue()).decode()

    # Note + log
    note = ("Runs completed. Thickness increases with temperature/time; wet grows faster. "
            "Uniformity is typically worse for wet. Consider a confirmation near target and a uniformity check.")
    uid = f"{req.student_id}-{int(time.time())}"

    with open("submissions_log.csv", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()},{req.student_id},{len(req.design.runs)},{json.dumps(req.target)}\n")

    return ExperimentResponse(
        run_log=f"Processed {len(req.design.runs)} run(s).",
        csv_base64=csv_b64,
        preview_plot_png_base64=png_b64,
        tech_note=note,
        uid=uid
    )
