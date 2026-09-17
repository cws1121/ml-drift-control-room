import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Literal

import numpy as np
from engine import dataset, drift, fingerprint, gate, metrics, train
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="ML Drift Control Room")
ROOT = Path(__file__).parent
train_x, train_y = dataset(3500, 101)
reference, _ = dataset(1500, 202)
baseline = train(train_x, train_y)
active = baseline
version = "baseline-v1"
run = None
candidate = None
audit_log = []
lock = RLock()


def log(action, **details):
    audit_log.append(
        {"at": datetime.now(timezone.utc).isoformat(), "action": action, **details}
    )
    del audit_log[:-100]


@app.middleware("http")
async def local_only(request: Request, call_next):
    if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
        return JSONResponse({"detail": "Local demo only"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Origin rejected"}, status_code=403)
    return await call_next(request)


class Simulation(BaseModel):
    scenario: Literal["normal", "sensor_shift", "concept_shift", "missing_sensor"] = (
        "normal"
    )
    severity: float = Field(default=1, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=1000000)
    labels: bool = True
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


class RunId(BaseModel):
    run_id: str


def require_run(run_id):
    if run is None or run["id"] != run_id:
        raise HTTPException(409, "Stale experiment. Run the scenario again.")


@app.post("/api/simulate")
def simulate(body: Simulation):
    global run, candidate
    with lock:
        x, y = dataset(1500, body.seed, body.scenario, body.severity)
        # Monitor only the training partition; gate/audit labels are never shown here.
        monitor_x, monitor_y = x[:900], y[:900]
        prob = active.predict_proba(monitor_x)[:, 1]
        result = {
            "id": uuid.uuid4().hex,
            "config": body.model_dump(),
            "version": version,
            "hash": fingerprint(x, y),
            "drift": drift(reference, monitor_x),
            "metrics": metrics(active, monitor_x, monitor_y, body.threshold)
            if body.labels
            else None,
            "predicted_positive": float((prob >= body.threshold).mean()),
            "mean_risk": float(prob.mean()),
            "partitions": {"train_monitor": 900, "gate": 300, "audit": 300},
            "examples": [
                {
                    "values": [
                        None if np.isnan(v) else round(float(v), 2) for v in row
                    ],
                    "risk": float(p),
                    "label": int(label) if body.labels else None,
                }
                for row, p, label in zip(monitor_x[:8], prob[:8], monitor_y[:8])
            ],
        }
        run = {
            "id": result["id"],
            "x": x,
            "y": y,
            "body": body,
            "result": result,
            "champion": active,
            "version": version,
        }
        candidate = None
        log(
            "experiment",
            run_id=run["id"],
            scenario=body.scenario,
            dataset_hash=result["hash"],
            version=version,
        )
        return result


@app.post("/api/train")
def challenger(body: RunId):
    global candidate
    with lock:
        require_run(body.run_id)
        if not run["body"].labels:
            raise HTTPException(
                422, "Labeled data is required to train and evaluate a challenger."
            )
        if candidate is not None:
            return candidate["report"]
        x, y = run["x"], run["y"]
        model = train(x[:900], y[:900])
        threshold = run["body"].threshold
        champion_metrics = metrics(run["champion"], x[900:1200], y[900:1200], threshold)
        challenger_metrics = metrics(model, x[900:1200], y[900:1200], threshold)
        report = {
            "id": uuid.uuid4().hex,
            "run_id": run["id"],
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "eligible": gate(champion_metrics, challenger_metrics),
            "audit": None,
            "promoted": False,
            "rule": "Gate F1 gain ≥ 0.02 and Brier increase ≤ 0.02",
            "version": "candidate-" + run["id"][:6],
        }
        candidate = {"model": model, "report": report}
        log("trained", candidate=report["version"], eligible=report["eligible"])
        return report


class CandidateId(RunId):
    candidate_id: str


@app.post("/api/promote")
def promote(body: CandidateId):
    global active, version
    with lock:
        require_run(body.run_id)
        if candidate is None or candidate["report"]["id"] != body.candidate_id:
            raise HTTPException(409, "Stale candidate")
        report = candidate["report"]
        if not report["eligible"]:
            raise HTTPException(422, "Promotion gate failed")
        if not report["promoted"]:
            # Audit is revealed only after selection, never used to tune the gate.
            x, y = run["x"][1200:], run["y"][1200:]
            t = run["body"].threshold
            report["audit"] = {
                "champion": metrics(run["champion"], x, y, t),
                "challenger": metrics(candidate["model"], x, y, t),
            }
            active = candidate["model"]
            version = report["version"]
            report["promoted"] = True
            log("promoted_locally", version=version, audit=report["audit"])
        return report


@app.post("/api/rollback")
def rollback():
    global active, version, run, candidate
    with lock:
        active = baseline
        version = "baseline-v1"
        run = None
        candidate = None
        log("rollback", version=version)
        return {"version": version}


@app.get("/api/status")
def status():
    with lock:
        return {
            "version": version,
            "audit_log": audit_log,
            "training_rows": 3500,
            "reference_rows": 1500,
        }


@app.get("/api/report")
def report():
    with lock:
        return JSONResponse(
            {
                "experiment": run["result"] if run else None,
                "candidate": candidate["report"] if candidate else None,
                "active_version": version,
                "events": audit_log,
                "scope": "Synthetic, local, session-only demonstration",
            },
            headers={
                "Content-Disposition": 'attachment; filename="ml-experiment.json"'
            },
        )


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
