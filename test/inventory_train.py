"""Kiểm kê train thành các bảng phẳng: tổng quan, app, tool, collection, trường.
Ghi data/derived/TRAIN_INVENTORY.md (không commit: dữ liệu BTC)."""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "data", "raw", "train+public_test", "train")
OUT = os.path.join(HERE, "..", "data", "derived", "TRAIN_INVENTORY.md")
cat = json.load(open(os.path.join(BASE, "train_tools_catalog.json"), encoding="utf-8"))
env_dir = os.path.join(BASE, "train_environment")
ids = sorted(os.listdir(env_dir))

def jtype(v):
    return {type(None): "null", bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}.get(type(v), type(v).__name__)

def short(v, n=60):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"

coll_samples = collections.Counter(); coll_items = collections.Counter(); coll_kind = {}
field_types = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
field_items = collections.defaultdict(collections.Counter); field_ex = collections.defaultdict(dict)
meta_keys = collections.Counter()

for i in ids:
    st = json.load(open(os.path.join(env_dir, i, "state.json"), encoding="utf-8"))
    meta_keys.update(st.get("_meta", {}).keys())
    for app, data in st.items():
        if app == "_meta" or not isinstance(data, dict): continue
        for coll, v in data.items():
            key = f"{app}.{coll}"
            coll_samples[key] += 1
            if isinstance(v, list):
                coll_kind[key] = "list"; coll_items[key] += len(v); records = [r if isinstance(r, dict) else {"(giá trị)": r} for r in v]
            elif isinstance(v, dict):
                coll_kind[key] = "dict"; coll_items[key] += len(v); records = [v]
            else:
                coll_kind[key] = "giá trị đơn"; coll_items[key] += 1; records = [{"(giá trị)": v}]
            for rec in records:
                for f, fv in rec.items():
                    field_types[key][f][jtype(fv)] += 1
                    field_items[key][f] += 1
                    field_ex[key].setdefault(f, fv)

# tool
tool_rows = []
for app in cat["apps"]:
    for t in app["tools"]:
        req = ", ".join(t["parameters"].get("required", [])) or "-"
        tool_rows.append((t["name"], t["kind"], req, t.get("description", "").strip().replace("|", "/")[:120]))

out = []
out.append("# Kiểm kê dữ liệu train\n")
out.append("> Tự sinh bởi `test/inventory_train.py`. Không commit: đây là dữ liệu BTC.\n")
out.append("## 1. Tổng quan\n")
out.append(f"- Mẫu train: **{len(ids)}**")
out.append(f"- App: **{len(cat['apps'])}**")
out.append(f"- Tool: **{len(tool_rows)}** (gồm 1 tool meta `sys.ask_missing_info`)")
out.append(f"- Collection trong `state.json`: **{len(coll_samples)}**")
out.append(f"- Khoá trong `_meta`: " + ", ".join(f"`{k}`" for k in meta_keys) + " (đều có ở mọi mẫu)\n")

out.append("## 2. App\n")
out.append("| app | tên | nhóm | tool | read | write | collection trong state |")
out.append("|---|---|---|---|---|---|---|")
for app in cat["apps"]:
    kinds = collections.Counter(t["kind"] for t in app["tools"])
    colls = ", ".join(c.split(".", 1)[1] for c in coll_samples if c.split(".", 1)[0] == app["id"]) or "-"
    out.append(f"| {app['id']} | {app.get('name_vi','')} | {app.get('group','')} | {len(app['tools'])} | {kinds.get('read',0)} | {kinds.get('write',0)} | {colls} |")

out.append("\n## 3. Tool\n")
out.append("| tool | kind | tham số bắt buộc | mô tả |")
out.append("|---|---|---|---|")
for name, kind, req, desc in tool_rows:
    out.append(f"| `{name}` | {kind} | {req} | {desc} |")

out.append("\n## 4. Collection trong state\n")
out.append("| collection | kiểu | số mẫu có | tổng phần tử | trường |")
out.append("|---|---|---|---|---|")
for key in sorted(coll_samples):
    fields = ", ".join(f"`{f}`" for f in field_types[key]) or "-"
    out.append(f"| `{key}` | {coll_kind[key]} | {coll_samples[key]} | {coll_items[key]} | {fields} |")

out.append("\n## 5. Chi tiết trường của collection\n")
out.append("Mỗi dòng là một trường. `số phần tử có trường` đếm trên mọi phần tử của collection (không phải số mẫu).\n")
out.append("| collection | trường | kiểu | số phần tử có trường | ví dụ |")
out.append("|---|---|---|---|---|")
for key in sorted(coll_samples):
    for f in sorted(field_types[key], key=lambda x: -field_items[key][x]):
        tstr = "/".join(sorted(field_types[key][f]))
        out.append(f"| `{key}` | `{f}` | {tstr} | {field_items[key][f]} | `{short(field_ex[key][f])}` |")

open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", OUT, "lines:", len(out))
