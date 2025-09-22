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

ALLOWED   = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
LABTOKEN  = os.getenv("LABTOKEN", "")

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
    uid: str

# ---------- Simulator ----------
def get_rng(student_id: str) -> random.Random:
    seed = int(hashlib.sha256((student_id + ASSIGNMENT_ID).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)

def growth_nm(temp_C: int, ambient: str, time_min: int) -> float:
    DRY = {900:0.6, 1000:1.0, 1100:3.3}
    WET = {900:2.0, 1000:5.0, 1100:10.0}
    base = (DRY if ambient == "dry" else WET)[temp_C]
    return base * time_min

# ---------- Routes ----------
@app.post("/run_experiments", response_model=ExperimentResponse)
def run_experiments(req: ExperimentRequest, x_labtoken: Optional[str] = Header(None)):
    check_token(x_labtoken)

    rng = get_rng(req.student_id)
    rows = [("run_id","temp_C","ambient","time_min","orientation","preclean",
             "thick_center_nm","thick_edge_nm","nonuniformity_pct")]
    centers = []

    for i, r in enumerate(req.design.runs, start=1):
        mean = growth_nm(r.temp_C, r.ambient, r.time_min)
        orient_k = 0.90 if r.orientation == "111" else 1.00
        pre_k    = 1.05 if r.preclean else 1.00
        base_noise_sigma = 0.04
        big_bump = 1.0 + abs(rng.gauss(0, 0.5)) * 0.01
        noise = rng.gauss(0, base_noise_sigma) * big_bump
        center = max(0.0, mean * orient_k * pre_k * (1 + noise))
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

    note = ("Runs completed. Thickness increases with temperature/time; wet grows faster
