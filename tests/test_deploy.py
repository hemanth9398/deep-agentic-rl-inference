"""Deployment rendering: every cross-reference between the generated objects must line up."""

import json
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from swarmrl.cli import main
from swarmrl.deploy import inference, training
from swarmrl.deploy.profile import DeployProfile, Rail
from swarmrl.deploy.render import make_plan, render, write

ROOT = Path(__file__).resolve().parents[1]
PROFILES = sorted((ROOT / "deploy" / "profiles").glob("*.yaml"))


def objects(files: dict) -> list[dict]:
    return [o for path, docs in files.items() if "/values/" not in path for o in docs]


def by_kind(objs, kind):
    return [o for o in objs if o["kind"] == kind]


def pod_specs(objs):
    for o in objs:
        if o["kind"] == "RayJob":
            rc = o["spec"]["rayClusterSpec"]
            yield o, rc["headGroupSpec"]["template"]
            for g in rc["workerGroupSpecs"]:
                yield o, g["template"]
        elif o["kind"] == "Deployment":
            yield o, o["spec"]["template"]


@pytest.fixture(params=[p.stem for p in PROFILES])
def rendered(request):
    p = DeployProfile.load(ROOT / "deploy" / "profiles" / f"{request.param}.yaml")
    return p, render(p)


def test_profile_loading_rejects_mistakes(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("name: x\nrolout: pool\n")
    with pytest.raises(ValueError, match="unknown profile keys"):
        DeployProfile.load(f)
    for bad in ({"link_type": "roce"}, {"scheduler": "slurm"}, {"rollout": "remote"}):
        with pytest.raises(ValueError):
            DeployProfile(**bad)
    f.write_text("rails: [{name: r0, pf: eth9}]\n")
    assert DeployProfile.load(f).rails == [Rail("r0", "eth9")]


def test_rdma_attachments_match_sriov_networks(rendered):
    p, files = rendered
    objs = objects(files)
    nets = {(o["metadata"]["name"], o["spec"]["networkNamespace"]): o["spec"]["resourceName"]
            for o in objs if o["kind"] in ("SriovNetwork", "SriovIBNetwork")}  # fmt: skip
    policies = {o["spec"]["resourceName"] for o in by_kind(objs, "SriovNetworkNodePolicy")}
    assert policies == {r.name for r in p.rails}
    for owner, tmpl in pod_specs(objs):
        ann = tmpl.get("metadata", {}).get("annotations", {}).get("k8s.v1.cni.cncf.io/networks")
        if not ann:
            continue
        ns = owner["metadata"]["namespace"]
        c = tmpl["spec"]["containers"][0]
        attached = json.loads(ann)
        for i, a in enumerate(attached):
            assert a["namespace"] == ns and a["interface"] == f"net{i + 1}"
            resource = nets[(a["name"], ns)]  # the NAD exists in the pod's namespace
            assert c["resources"]["limits"][f"openshift.io/{resource}"] == "1"
        assert "IPC_LOCK" in c["securityContext"]["capabilities"]["add"]
        assert any(m["mountPath"] == "/dev/shm" for m in c["volumeMounts"])
        env = {e["name"]: e.get("value") for e in c["env"]}
        assert env["NCCL_SOCKET_IFNAME"] == "eth0"
        assert ("NCCL_IB_GID_INDEX" in env) == (p.link_type == "eth" and p.roce_gid_index is not None)


def test_scheduler_references_resolve(rendered):
    p, files = rendered
    objs = objects(files)
    if p.scheduler == "kueue":
        lqs = {(o["metadata"]["namespace"], o["metadata"]["name"]): o["spec"]["clusterQueue"]
               for o in by_kind(objs, "LocalQueue")}  # fmt: skip
        cqs = {o["metadata"]["name"] for o in by_kind(objs, "ClusterQueue")}
        prios = {o["metadata"]["name"] for o in by_kind(objs, "WorkloadPriorityClass")}
        topo = by_kind(objs, "Topology")[0]
        levels = [lvl["nodeLabel"] for lvl in topo["spec"]["levels"]]
        assert by_kind(objs, "ResourceFlavor")[0]["spec"]["topologyName"] == topo["metadata"]["name"]
        assert set(lqs.values()) <= cqs
        for o in objs:
            labels = o["metadata"].get("labels", {})
            if "kueue.x-k8s.io/queue-name" in labels:
                assert (o["metadata"]["namespace"], labels["kueue.x-k8s.io/queue-name"]) in lqs
                assert labels["kueue.x-k8s.io/priority-class"] in prios
        for _, tmpl in pod_specs(objs):
            t = tmpl.get("metadata", {}).get("annotations", {}).get("kueue.x-k8s.io/podset-required-topology")
            assert t is None or t in levels
            assert "schedulerName" not in tmpl["spec"]
        assert files["20-scheduling/values/kueue-manager-config.yaml"]["resources"]["excludeResourcePrefixes"] == [
            "openshift.io/"
        ]
    else:
        assert not any("kueue.x-k8s.io/queue-name" in o["metadata"].get("labels", {}) for o in objs)
        queues = {o["metadata"]["name"] for o in by_kind(objs, "Queue")}
        hns = by_kind(objs, "HyperNode")
        assert {h["metadata"]["name"] for h in hns if h["spec"]["tier"] == 1} == {f"rack-{r}" for r in p.racks}
        for owner, tmpl in pod_specs(objs):
            assert tmpl["spec"]["schedulerName"] == "volcano"
            if owner["kind"] == "RayJob":
                assert owner["metadata"]["labels"]["volcano.sh/queue-name"] in queues


def test_training_job_wiring(rendered):
    p, files = rendered
    objs = objects(files)
    (job,) = by_kind(objs, "RayJob")
    sa = {o["metadata"]["name"] for o in by_kind(objs, "ServiceAccount")}
    (cm,) = [o for o in by_kind(objs, "ConfigMap") if o["metadata"]["name"] == "swarmrl-slime"]
    settings = json.loads(cm["data"]["slime.json"])
    entry = job["spec"]["entrypoint"]
    assert "--custom-generate-function-path swarmrl.integrations.slime.generate" in entry
    assert "--custom-reward-post-process-path swarmrl.integrations.slime.reward_post_process" in entry
    assert settings["namespace"] == p.sandbox_namespace and settings["sandbox"] == "k8s"
    runtime_env = json.loads(job["spec"]["runtimeEnvYAML"])["env_vars"]
    assert runtime_env["SWARMRL_SLIME_CONFIG"] == training.CONFIG_PATH
    for _, tmpl in pod_specs([job]):
        spec = tmpl["spec"]
        assert spec["serviceAccountName"] in sa
        assert any(v.get("configMap", {}).get("name") == cm["metadata"]["name"] for v in spec["volumes"])
    workers = job["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert workers["replicas"] == p.train_nodes
    gpus = workers["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]
    assert gpus == str(p.gpus_per_node)
    role_ns = {o["metadata"]["namespace"] for o in by_kind(objs, "Role")}
    assert p.sandbox_namespace in role_ns
    netpols = [o for o in by_kind(objs, "NetworkPolicy") if o["metadata"]["namespace"] == p.sandbox_namespace]
    assert netpols and set(netpols[0]["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    if p.rollout == "pool":
        assert (
            entry.startswith("ENGINES=$(swarmrl pool-endpoints") and "--rollout-external-engine-addrs $ENGINES" in entry
        )
        assert "--colocate" not in entry and settings["policy_url"] == training.gateway_url(p)
        assert p.inference_namespace in role_ns  # pool discovery needs to list pods there
    else:
        assert "--colocate" in entry and "--rollout-external-engine-addrs" not in entry


def test_rollout_pool_and_epp_agree(rendered):
    p, files = rendered
    if p.rollout != "pool":
        assert "40-inference/rollout-pool.yaml" not in files
        return
    (pool,) = files["40-inference/rollout-pool.yaml"]
    values = files["40-inference/values/llm-d-router.yaml"]
    pl = make_plan(p)
    assert pool["spec"]["replicas"] * pl.rollout.engine_tp == p.pool_gpus
    labels = pool["spec"]["template"]["metadata"]["labels"]
    assert values["router"]["modelServers"]["matchLabels"].items() <= labels.items()
    c = pool["spec"]["template"]["spec"]["containers"][0]
    args = c["args"]
    epp = yaml.safe_load(values["router"]["epp"]["pluginsCustomConfig"][values["router"]["epp"]["pluginsConfigFile"]])
    plugins = {pl_.get("name", pl_["type"]): pl_ for pl_ in epp["plugins"]}
    refs = [r["pluginRef"] for prof in epp["schedulingProfiles"] for r in prof["plugins"]]
    refs += [r["pluginRef"] for r in epp["requestHandler"]["parsers"]]
    refs += [x["pluginRef"] for s in epp["dataLayer"]["sources"] for x in [s, *s["extractors"]]]
    assert set(refs) <= set(plugins)
    producer = plugins["precise-prefix-cache-producer"]["parameters"]
    assert producer["kvEventsConfig"]["engineType"] == "sglang"
    assert str(producer["tokenProcessorConfig"]["blockSizeTokens"]) == args[args.index("--page-size") + 1]
    port = producer["kvEventsConfig"]["podDiscoveryConfig"]["socketPort"]
    assert {"containerPort": port, "name": "kv-events"} in c["ports"]
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert env["KV_EVENTS_ENDPOINT"] == f"tcp://*:{port}"
    kv = json.loads(args[args.index("--kv-events-config") + 1])
    assert kv["topic"].startswith(producer["kvEventsConfig"]["topicFilter"])
    assert values["router"]["modelServers"]["targetPorts"] == [{"number": training.POOL_PORT}]
    assert values["httpRoute"]["inferenceGatewayName"] == by_kind(objects(files), "Gateway")[0]["metadata"]["name"]


def test_serving_uses_the_vllm_epp(rendered):
    p, files = rendered
    (svc,) = files["50-serving/llminferenceservice.yaml"]
    assert svc["apiVersion"] == "serving.kserve.io/v1alpha2"
    inline = svc["spec"]["router"]["scheduler"]["config"]["inline"]
    assert {"type": "vllmhttp-parser"} in inline["plugins"]
    env = {e["name"]: e.get("value") for e in svc["spec"]["template"]["containers"][0]["env"]}
    assert f"--block-size {inference.BLOCK_TOKENS}" in env["VLLM_ADDITIONAL_ARGS"]


def test_infiniband_profile_uses_ib_networks():
    p = DeployProfile(link_type="ib")
    objs = objects(render(p))
    assert by_kind(objs, "SriovIBNetwork") and not by_kind(objs, "SriovNetwork")
    assert all(o["spec"]["linkType"] == "ib" for o in by_kind(objs, "SriovNetworkNodePolicy"))


def test_committed_manifests_are_up_to_date(tmp_path):
    """deploy/rendered is what reviewers read: it must be exactly what the code renders."""
    for prof in PROFILES:
        out = tmp_path / prof.stem
        write(DeployProfile.load(prof), out)
        committed = ROOT / "deploy" / "rendered" / prof.stem
        got = {f.relative_to(out): f.read_text() for f in out.rglob("*") if f.is_file()}
        want = {f.relative_to(committed): f.read_text() for f in committed.rglob("*") if f.is_file()}
        assert got == want, f"re-render deploy/rendered/{prof.stem} (swarmrl deploy --profile {prof} --out ...)"


def test_cli_deploy_and_pool_endpoints(tmp_path, monkeypatch, capsys):
    assert main(["deploy", "--profile", str(PROFILES[0]), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "README.md").exists()

    def pod(name, ip, ready, deleting=False):
        meta = {"name": name, **({"deletionTimestamp": "x"} if deleting else {})}
        return {"metadata": meta, "status": {"podIP": ip, "conditions": [{"type": "Ready", "status": ready}]}}

    pods = {"items": [pod("a", "10.0.0.2", "True"), pod("b", "10.0.0.1", "True"), pod("c", "10.0.0.3", "False"),
                      pod("d", "10.0.0.4", "True", deleting=True)]}  # fmt: skip
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps(pods), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    capsys.readouterr()
    assert main(["pool-endpoints", "--namespace", "inference", "--selector", "app=x"]) == 0
    assert capsys.readouterr().out.strip() == "10.0.0.1:8000 10.0.0.2:8000"
    assert calls[0][:4] == ["kubectl", "get", "pods", "-n"]
    assert main(["pool-endpoints", "--namespace", "inference", "--selector", "app=x", "--min-ready", "3"]) == 1
