import app
from engine import dataset, drift, fingerprint, gate, metrics, train
from fastapi.testclient import TestClient

client = TestClient(app.app)


def reset():
    client.post("/api/rollback")


def test_reproducibility_and_disjoint_partitions():
    x, y = dataset(1500, 42)
    xx, yy = dataset(1500, 42)
    assert fingerprint(x, y) == fingerprint(xx, yy)
    ids = [set(range(900)), set(range(900, 1200)), set(range(1200, 1500))]
    assert not ids[0] & ids[1] and not ids[1] & ids[2] and not ids[0] & ids[2]
    assert fingerprint(x[:900], y[:900]) != fingerprint(x[900:1200], y[900:1200])


def test_concept_shift_can_escape_input_monitoring():
    x, y = dataset(1500, 42, "concept_shift")
    assert sum(f["alert"] for f in drift(app.reference, x[:900])) == 0
    candidate = train(x[:900], y[:900])
    old = metrics(app.baseline, x[900:1200], y[900:1200])
    new = metrics(candidate, x[900:1200], y[900:1200])
    assert new["f1"] > old["f1"] + 0.3 and gate(old, new)


def test_sensor_and_missingness_alerts():
    assert (
        sum(
            f["alert"]
            for f in drift(app.reference, dataset(1500, 42, "sensor_shift")[0])
        )
        >= 2
    )
    missing = drift(app.reference, dataset(1500, 42, "missing_sensor")[0])[1]
    assert missing["alert"] and missing["missing"] > 0.5


def test_hidden_labels_block_training():
    reset()
    r = client.post("/api/simulate", json={"labels": False}).json()
    assert r["metrics"] is None and all(e["label"] is None for e in r["examples"])
    assert client.post("/api/train", json={"run_id": r["id"]}).status_code == 422


def test_promotion_audit_idempotency_and_stale_candidate():
    reset()
    r = client.post("/api/simulate", json={"scenario": "concept_shift"}).json()
    c = client.post("/api/train", json={"run_id": r["id"]}).json()
    assert c["eligible"] and c["audit"] is None
    payload = {"run_id": r["id"], "candidate_id": c["id"]}
    promoted = client.post("/api/promote", json=payload).json()
    assert promoted["promoted"] and promoted["audit"]["challenger"]["n"] == 300
    assert client.post("/api/promote", json=payload).json() == promoted
    assert client.get("/api/status").json()["version"] == c["version"]
    client.post("/api/simulate", json={})
    assert client.post("/api/promote", json=payload).status_code == 409
    reset()
    assert client.get("/api/status").json()["version"] == "baseline-v1"


def test_gate_rejects_regressions():
    assert not gate({"f1": 0.8, "brier": 0.1}, {"f1": 0.81, "brier": 0.1})
    assert not gate({"f1": 0.8, "brier": 0.1}, {"f1": 0.9, "brier": 0.15})


def test_validation_and_origin():
    assert client.post("/api/simulate", json={"severity": 2}).status_code == 422
    assert (
        client.post(
            "/api/rollback", headers={"Origin": "https://example.com"}
        ).status_code
        == 403
    )
