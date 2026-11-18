"""Fleet report logic on recorded cluster state (no cluster needed)."""
import fleet_report
import whats_new


def node(gpu="1", product="NVIDIA-A100-80GB-PCIe", itype="Standard_NC24ads_A100_v4", driver="595.91.07"):
    return {"metadata": {"labels": {"plv.io/pool": "gpu", "nvidia.com/gpu.product": product, "node.kubernetes.io/instance-type": itype,
                                    "nvidia.com/cuda.driver-version.full": driver}},
            "status": {"allocatable": {"nvidia.com/gpu": gpu}}}


def raw(nim_tag="2.0.13", canary="", weight="0", app_health="Healthy", restarts=0):
    api_env = [{"name": "BASE_MODEL", "value": "meta/llama-3.1-8b-instruct"}, {"name": "STABLE_MODEL", "value": ""},
               {"name": "CANARY_MODEL", "value": canary}, {"name": "CANARY_WEIGHT", "value": weight}]
    return {"nodes": {"items": [node(), {"metadata": {"labels": {}}, "status": {"allocatable": {}}}]},
            "apps": {"items": [{"metadata": {"name": "plover"}, "status": {"sync": {"status": "Synced"}, "health": {"status": app_health}}}]},
            "nims": {"items": [{"metadata": {"name": "plover-llm"}, "spec": {"image": {"repository": "nvcr.io/nim/meta/llama-3.1-8b-instruct", "tag": nim_tag}},
                                "status": {"state": "Ready"}}]},
            "deploys": {"items": [{"metadata": {"name": "plover-api"}, "spec": {"template": {"spec": {"containers": [{"env": api_env}]}}}}]},
            "pods": {"items": [{"metadata": {"name": "plover-llm-0"}, "status": {"containerStatuses": [{"restartCount": restarts}]}}]}}


def test_green_cluster_has_no_attention_items():
    s = fleet_report.analyze("aks", raw(), 0.68, fleet_report.pins())
    assert s["gpu_nodes"] == 1 and s["gpus"] == 1 and s["usd_per_hour"] == 0.68
    assert s["attention"] == []
    md = fleet_report.render([s], "now")
    assert "Nothing. Every cluster is synced" in md and "| aks |" in md


def test_drift_canary_and_crashloops_are_flagged():
    s = fleet_report.analyze("gke", raw(nim_tag="2.0.12", canary="plover-style-v2", weight="20", app_health="Degraded", restarts=7), 1.63, fleet_report.pins())
    text = " ".join(s["attention"])
    assert "differs from the pin" in text and "canary plover-style-v2 at 20%" in text
    assert "Degraded" in text and "restarting" in text


def test_whats_new_status_math():
    assert whats_new.status("v26.7.2", "v26.7.3") == "patch behind"
    assert whats_new.status("3.1.2", "3.2.0") == "minor behind"
    assert whats_new.status("v1.36.5+rke2r1", "v1.37.1+rke2r1") == "minor behind"
    assert whats_new.status("0.13.0", "0.13.0") == "current"
    assert whats_new.newest(["v1.0.0", "v1.10.0", "v1.9.9", "v2.0.0-rc1"]) == "v1.10.0"
