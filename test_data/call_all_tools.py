"""Gọi thử mọi tool trong catalog của mỗi split bằng zaic_tools (không sửa mô phỏng).

Với mỗi tool: tìm một mẫu có app của tool được cài, sinh tham số tối thiểu theo schema,
rồi gọi qua ToolEnv. Ghi kết quả ra runs/call_all_tools/results_<split>.jsonl và in tổng hợp.
Chạy: python test/call_all_tools.py
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "raw", "train+public_test")
DATE, DT, TM = "2026-11-10", "2026-11-10T09:00:00", "09:00"


def sample_value(prop, state_app, app_state):
    if "x-ref" in prop:  # id lấy từ dữ liệu app: thử lấy phần tử đầu tiên của collection
        coll = prop["x-ref"].split(".", 1)[-1]
        items = (app_state or {}).get(coll) or []
        if items and isinstance(items[0], dict) and "id" in items[0]:
            return items[0]["id"]
        return "test_id"
    if "enum" in prop:
        return prop["enum"][0]
    t = prop.get("type")
    fmt = prop.get("format")
    if t == "string":
        if fmt == "date": return DATE
        if fmt == "date-time": return DT
        if fmt == "time": return TM
        return "test"
    if t in ("integer", "number"):
        return prop.get("minimum", 1)
    if t == "boolean": return True
    if t == "array":
        it = prop.get("items", {})
        if "enum" in it: return [it["enum"][0]]
        if it.get("type") == "object": return [{}]
        if it.get("type") in ("integer", "number"): return [1]
        return ["test"] if it.get("format") is None else [sample_value(it, state_app, app_state)]
    if t == "object": return {}
    return "test"


def build_args(tool, app_state):
    p = tool.get("parameters", {})
    props = p.get("properties", {})
    args = {}
    for k in p.get("required", []):
        args[k] = sample_value(props.get(k, {}), tool["app"], app_state)
    if "x-one-of" in p and p["x-one-of"]:
        for k in p["x-one-of"][0]:
            args.setdefault(k, sample_value(props.get(k, {}), tool["app"], app_state))
    return args


def run_split(split):
    from zaic_tools import ToolEnv  # import theo đường dẫn của split
    base = os.path.join(DATA, split)
    cat = json.load(open(os.path.join(base, f"{split}_tools_catalog.json"), encoding="utf-8"))
    samples = [json.loads(l) for l in open(os.path.join(base, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
    tools = [t for a in cat["apps"] for t in a["tools"]]
    results = []
    for tool in tools:
        name = tool["name"]; app = tool["app"]
        # chọn mẫu có app được cài và tool nằm trong danh sách tool của mẫu
        pick = None
        for s in samples:
            inst = s["context"].get("installed_apps", {})
            if app in inst and name in inst[app].get("tools", []):
                pick = s; break
        if pick is None and name != "sys.ask_missing_info":
            results.append({"tool": name, "status": "no_sample"}); continue
        if pick is None:
            pick = samples[0]
        env = ToolEnv(split, pick["id"], root=DATA)
        app_state = env.get_state().get(app) if hasattr(env, "get_state") else None
        args = build_args(tool, app_state)
        if name == "sys.ask_missing_info":
            args = {"tool_name": "sms.send", "params": ["contact_id"]}
        try:
            res = env.call(name, args)
            ok = res.get("ok")
            status = "ok" if ok else "schema_rejected"
            err = res.get("error")
        except Exception as e:  # không nên xảy ra: mô phỏng không raise
            status, err = "exception", f"{type(e).__name__}: {e}"
        results.append({"tool": name, "kind": tool["kind"], "sample": pick["id"], "args": args, "status": status, "error": err,
                        "_line": {"id": pick["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}})
    OUTDIR = os.path.join(HERE, "..", "runs", "call_all_tools"); os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"results_{split}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps({k: v for k, v in r.items() if k != "_line"}, ensure_ascii=False) + "\n")
    # file nộp theo README mục 4: id, tool_calls, state_hash (mỗi dòng là một lần thử tool)
    sub = os.path.join(OUTDIR, f"submission_{split}.jsonl")
    with open(sub, "w", encoding="utf-8") as f:
        for r in results:
            line = r["_line"]
            line["tool_calls"] = [{"name": c["name"], "arguments": c["arguments"]} for c in line["tool_calls"]]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"  submission -> {sub}")
    c = collections.Counter(r["status"] for r in results)
    print(f"[{split}] tools={len(tools)} status={dict(c)} -> {out}")
    for r in results:
        if r["status"] != "ok":
            print("  ", r["status"], r["tool"], "|", r.get("error"), "| args:", r.get("args"))


if __name__ == "__main__":
    for split in ["train", "public_test"]:
        sys.path.insert(0, os.path.join(DATA, split))
        for m in [k for k in list(sys.modules) if k == "zaic_tools" or k.startswith("zaic_tools.")]:
            del sys.modules[m]
        run_split(split)
        sys.path.pop(0)
