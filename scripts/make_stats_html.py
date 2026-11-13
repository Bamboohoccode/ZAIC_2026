"""Sinh 2 trang thống kê HTML tự chứa (không cần mạng) vào data/derived/.
  a) stats_btc_20261111.html      : dữ liệu của BTC (catalog train + public_test, VERSION, mẫu train)
  b) stats_sft_multiturn_20261111.html : bộ data nhóm (sft_multiturn train + dev)
Chỉ đọc catalog của public_test và file VERSION; không đọc public_test.jsonl, environment hay nhãn test.
Chạy: python -X utf8 scripts/make_stats_html.py [--date 20261111]
"""
import argparse, collections, html, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/train+public_test"
OUT = ROOT / "data/derived"
E = html.escape

CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--surface:#fcfcfb;--panel:#f4f3f0;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e2e1dc;
--read:#2a78d6;--write:#eb6834;--meta:#1baf7a;--one:#2a78d6;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#1a1a19;--surface:#1a1a19;--panel:#242422;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8d8c85;--grid:#33332f;--read:#3987e5;--write:#d95926;--meta:#199e70;--one:#3987e5;}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--surface:#1a1a19;--panel:#242422;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8d8c85;--grid:#33332f;--read:#3987e5;--write:#d95926;--meta:#199e70;--one:#3987e5;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 6px;padding-top:10px;border-top:1px solid var(--grid)}h3{font-size:15px;margin:22px 0 4px}
p.sub{color:var(--ink2);margin:0 0 6px}p.note{color:var(--ink2);font-size:13px;margin:4px 0 10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.kpi{background:var(--panel);border-radius:10px;padding:10px 12px}.kpi b{display:block;font-size:24px;line-height:1.15}.kpi span{color:var(--ink2);font-size:12.5px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:18px}
.card{background:var(--panel);border-radius:12px;padding:12px 12px 8px;overflow:hidden}.card h3{margin-top:2px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:2px 0 6px}.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--ink2);font-size:11.5px}svg text.b{fill:var(--ink);font-weight:600}svg text.v{fill:var(--ink)}svg .gl{stroke:var(--grid);stroke-width:1}
.read{fill:var(--read);background:var(--read)}.write{fill:var(--write);background:var(--write)}.meta{fill:var(--meta);background:var(--meta)}.one{fill:var(--one);background:var(--one)}
svg rect.m{stroke:var(--panel);stroke-width:2}svg rect.m:hover{opacity:.8}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:left}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
details{margin:10px 0}summary{cursor:pointer;color:var(--ink2)}code{background:var(--panel);padding:1px 5px;border-radius:4px;font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:5px 9px;border-radius:6px;font-size:12.5px;display:none;z-index:9;max-width:320px;white-space:pre-line}
.tag{display:inline-block;background:var(--panel);border-radius:999px;padding:1px 9px;margin:2px 3px 2px 0;font-size:12.5px}
@media(max-width:560px){main{padding:18px 16px 50px}h1{font-size:22px}}
"""
JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');if(!t){tip.style.display='none';return}
tip.textContent=t.getAttribute('data-tip');tip.style.display='block';
const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+16)+'px'});
"""


def page(title, sub, body):
    return (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{E(title)}</title><style>{CSS}</style></head><body><main><h1>{E(title)}</h1><p class=\"sub\">{sub}</p>{body}</main>"
            f'<div id="tip"></div><script>{JS}</script></body></html>')


def kpis(items):
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{E(str(v))}</b><span>{E(k)}</span></div>' for k, v in items) + "</div>"


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i class="{c}"></i>{E(n)}</span>' for n, c in items) + "</div>"


def hbar(rows, classes, label_w=150, width=520, bar_h=15, gap=6, names=None):
    """rows: {'header':str} hoặc {'label','vals','tip'}. vals xếp chồng theo classes."""
    mx = max([sum(r["vals"]) for r in rows if "vals" in r] + [1])
    plot_w = width - label_w - 40
    y = 4; out = []
    for r in rows:
        if "header" in r:
            y += 6; out.append(f'<text class="b" x="0" y="{y+11}">{E(r["header"])}</text>'); y += 20; continue
        tot = sum(r["vals"]); x = label_w
        out.append(f'<text x="{label_w-8}" y="{y+bar_h-3}" text-anchor="end">{E(r["label"])}</text>')
        for v, c in zip(r["vals"], classes):
            if v <= 0: continue
            w = max(v / mx * plot_w, 3)
            out.append(f'<rect class="m {c}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" data-tip="{E(r.get("tip", r["label"]))}"/>')
            x += w
        out.append(f'<text class="v" x="{x+5:.1f}" y="{y+bar_h-3}">{tot}</text>')
        y += bar_h + gap
    return f'<svg viewBox="0 0 {width} {y+4}" role="img">' + "".join(out) + "</svg>"


def vbar(labels, values, cls="one", width=520, height=190, tips=None, fmt=str):
    mx = max(values + [1]); pl, pb, pt = 8, 36, 18
    n = len(values); bw = (width - 2 * pl) / n; ph = height - pb - pt
    out = [f'<line class="gl" x1="{pl}" x2="{width-pl}" y1="{pt+ph}" y2="{pt+ph}"/>']
    for i, (lb, v) in enumerate(zip(labels, values)):
        h = v / mx * ph; x = pl + i * bw + bw * 0.14; w = bw * 0.72
        tp = tips[i] if tips else f"{lb}: {v}"
        out.append(f'<rect class="m {cls}" x="{x:.1f}" y="{pt+ph-h:.1f}" width="{w:.1f}" height="{max(h,1):.1f}" rx="3" data-tip="{E(tp)}"/>')
        out.append(f'<text class="v" x="{x+w/2:.1f}" y="{pt+ph-h-4:.1f}" text-anchor="middle">{fmt(v)}</text>')
        out.append(f'<text x="{x+w/2:.1f}" y="{pt+ph+15}" text-anchor="middle">{E(str(lb))}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">' + "".join(out) + "</svg>"


def table(head, rows, num_cols=()):
    th = "".join(f'<th class="{"n" if i in num_cols else ""}">{E(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in num_cols else ""}">{E(str(c))}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{tr}</table>"


def card(title, inner, note=""):
    return f'<div class="card"><h3>{E(title)}</h3>{inner}{f"<p class=note>{E(note)}</p>" if note else ""}</div>'


KIND = [("read (chỉ đọc)", "read"), ("write (thay đổi dữ liệu)", "write"), ("meta (hỏi lại thông tin)", "meta")]


KIND_DEF = {
    "read": "chỉ đọc dữ liệu, không đổi trạng thái của app (ví dụ: xem danh sách, xem giá).",
    "write": "đổi trạng thái của app hoặc tạo hành động (ví dụ: đặt báo thức, gửi tin, chuyển tiền).",
    "meta": "tool hệ thống, không chạy trên trạng thái app. Hiện chỉ có sys.ask_missing_info: báo lại khi thiếu tham số bắt buộc để hỏi người dùng.",
}


def load_catalog(split):
    c = json.load(open(RAW / split / f"{split}_tools_catalog.json", encoding="utf-8"))
    apps = {}
    for a in c["apps"]:
        k = collections.Counter(t["kind"] for t in a["tools"])
        apps[a["id"]] = {"name": a["name_vi"], "group": a["group"], "desc": a.get("description", ""), "tools": [t["name"] for t in a["tools"]],
                         "kind": {t["name"]: t["kind"] for t in a["tools"]},
                         "tool_info": {t["name"]: {"kind": t["kind"], "desc": t.get("description", ""), "required": t.get("parameters", {}).get("required", [])} for t in a["tools"]},
                         "read": k["read"], "write": k["write"], "meta": k["meta"]}
    return apps


def formula(apps):
    """Chuỗi tổng: số tool = read + write + meta, kèm thẻ định nghĩa ba loại."""
    k = collections.Counter(kind for a in apps.values() for kind in a["kind"].values())
    total = sum(k.values())
    assert total == k["read"] + k["write"] + k["meta"]
    eq = f"{total} = {k['read']} read + {k['write']} write + {k['meta']} meta"
    cards = "".join(f'<div class="kpi"><b>{E(kind)}: {k[kind]}</b><span>{E(KIND_DEF[kind])}</span></div>' for kind in ("read", "write", "meta"))
    return f'<p class=note><b>Tổng tool: {E(eq)}.</b> Mỗi tool thuộc đúng một loại, nên ba số này cộng lại bằng tổng.</p><div class="kpis">{cards}</div>'


def tr_kind(apps, tool):
    app = tool.split(".", 1)[0]
    return apps[app]["kind"][tool]


def app_detail(aid, a):
    rows = []
    for t in a["tools"]:
        ti = a["tool_info"][t]
        req = ", ".join(ti["required"]) or "không"
        rows.append([t, ti["kind"], ti["desc"], req])
    return (f'<details><summary><b>{E(a["name"])}</b> (<code>{E(aid)}</code>): {len(a["tools"])} tool. {E(a["desc"])}</summary>'
            + table(["tool", "loại", "tool làm gì", "tham số bắt buộc"], rows) + "</details>")


def group_table(apps):
    g = collections.defaultdict(lambda: {"apps": [], "read": 0, "write": 0, "meta": 0})
    for aid, a in apps.items():
        d = g[a["group"]]; d["apps"].append(aid)
        for k in ("read", "write", "meta"): d[k] += a[k]
    return g


def group_rows(apps):
    g = group_table(apps)
    rows = []
    for name, d in sorted(g.items(), key=lambda kv: -(kv[1]["read"] + kv[1]["write"] + kv[1]["meta"])):
        t = d["read"] + d["write"] + d["meta"]
        rows.append({"label": f"{name} ({len(d['apps'])} app)", "vals": [d["read"], d["write"], d["meta"]],
                     "tip": f"nhóm {name}: {len(d['apps'])} app, {t} tool\nread {d['read']}, write {d['write']}, meta {d['meta']}\napp: " + "; ".join(f"{x} ({apps[x]['name']})" for x in d["apps"])})
    return rows, g


def app_rows(apps, g):
    rows = []
    for name, d in sorted(g.items(), key=lambda kv: -(kv[1]["read"] + kv[1]["write"] + kv[1]["meta"])):
        t = d["read"] + d["write"] + d["meta"]
        rows.append({"header": f"{name}: {len(d['apps'])} app, {t} tool"})
        for aid in sorted(d["apps"], key=lambda x: -len(apps[x]["tools"])):
            a = apps[aid]
            rows.append({"label": aid, "vals": [a["read"], a["write"], a["meta"]],
                         "tip": f"{aid} ({a['name']})\n{len(a['tools'])} tool: read {a['read']}, write {a['write']}, meta {a['meta']}"})
    return rows


def app_table_rows(apps):
    return [[a["group"], aid, a["name"], len(a["tools"]), a["read"], a["write"], a["meta"]]
            for aid, a in sorted(apps.items(), key=lambda kv: (kv[1]["group"], kv[0]))]


APP_HEAD = ["nhóm", "app", "tên", "tool", "read", "write", "meta"]


def hist(values, bins=None):
    c = collections.Counter(values)
    ks = sorted(c)
    return [str(k) for k in ks], [c[k] for k in ks]


# ---------------------------------------------------------------- file 1: BTC
SFT_EXAMPLE = """{
  "id": "train_0454", "split": "train", "source": "btc", "category": "act",
  "apps": ["bill_payment", "calendar", "contacts", "ewallet", "notes", "subscriptions"],
  "n_calls": 2, "n_history": 2, "n_tokens": 3179,
  "messages": [
    {"role": "system", "content": "...vai trò, thời điểm hiện tại, vị trí, hồ sơ người dùng, app cài...", "train": false},
    ... (2 lượt history của BTC) ...
    {"role": "user", "content": "Ừ thế chuyển luôn số đấy cho Tuấn qua ví giúp anh.", "train": false},
    {"role": "assistant", "content": "", "train": true,
     "tool_calls": [{"id": "call_train_0454_0", "type": "function",
                     "function": {"name": "contacts.search", "arguments": {"query": "Tuấn"}}}]},
    {"role": "tool", "tool_call_id": "call_train_0454_0", "name": "contacts.search", "train": false,
     "content": "{\\"ok\\": true, \\"contacts\\": [{\\"id\\": \\"c_07\\", \\"name\\": \\"Tuấn\\", ...}]}"},
    {"role": "assistant", "content": "", "train": true,
     "tool_calls": [{"id": "call_train_0454_1", "type": "function",
                     "function": {"name": "ewallet.transfer", "arguments": {"contact_id": "c_07", "amount": 180000}}}]}
  ],
  "tools": [
    {"type": "function", "function": {"name": "calendar.list_events", "description": "Danh sách sự kiện, lọc được theo khoảng ngày",
                                      "parameters": {"type": "object", "properties": {"date_from": {...}, "date_to": {...}}, "required": []}}},
    ... (20 tool, gồm sys.ask_missing_info) ...
  ]
}"""

SFT_FIELDS_SAMPLE = [
    ("id", "mã mẫu, trùng id trong train.jsonl của BTC và tên thư mục train_environment/<id>/"),
    ("split", "train hoặc dev, theo eval/splits/dev_ids.txt"),
    ("source", "nguồn mẫu; hiện chỉ có btc (nhãn BTC, không sinh thêm)"),
    ("category", "act = gọi tool; ask = kết thúc bằng sys.ask_missing_info (hỏi lại thiếu tham số); empty = không gọi tool (yêu cầu ngoài các app đã cài)"),
    ("apps", "các app được cài ở mẫu này (context.installed_apps), đã sắp xếp. Chỉ tool của các app này được gọi"),
    ("n_calls", "số tool call gold ở lượt user cuối (không tính history)"),
    ("n_history", "số lượt (user và assistant) trong history trước yêu cầu cuối"),
    ("n_tokens", "độ dài mẫu theo chat template của tokenizer Qwen3.8-4B-Distilled, tính cả tools. Dùng để lọc mẫu quá dài"),
    ("messages", "hội thoại đầy đủ: 1 lượt system, các lượt history, 1 lượt user (yêu cầu cuối), rồi assistant (tool call) xen kẽ tool (kết quả)"),
    ("tools", "schema các tool được phép gọi: mọi tool của các app trong apps, cộng sys.ask_missing_info"),
]
SFT_FIELDS_MSG = [
    ("role", "system, user, assistant hoặc tool"),
    ("content", "system: lời dặn vai trò, thời điểm, vị trí, hồ sơ, app cài. user: lời nói đã chuyển thành chữ (audio_text). assistant: câu trả lời bằng lời (rỗng khi chỉ gọi tool). tool: kết quả tool dạng chuỗi JSON"),
    ("tool_calls", "chỉ ở assistant có gọi tool: danh sách {id, type: function, function: {name, arguments}}. arguments là dict, không phải chuỗi JSON"),
    ("tool_call_id", "chỉ ở tool: khớp id của tool call mà lượt này trả lời"),
    ("name", "chỉ ở tool: tên tool đã gọi, ví dụ contacts.search"),
    ("train", "true nếu là đích học (chỉ các lượt assistant gọi tool ở yêu cầu cuối); false cho system, history, user, tool result. Là mask loss gợi ý, tuỳ trainer"),
]
SFT_FIELDS_TOOL = [
    ("type", "luôn là function"),
    ("function.name", "tên tool dạng <app>.<tool>, ví dụ calendar.list_events"),
    ("function.description", "mô tả tool bằng tiếng Việt, lấy từ catalog BTC"),
    ("function.parameters", "JSON Schema của arguments: properties, required. Giữ nguyên mở rộng của BTC: x-ref (id lấy từ dữ liệu app khác), x-one-of (bắt buộc một trong các nhóm), x-match: fuzzy (văn bản tự do, chấm gần đúng)"),
]


def sft_vs_raw_section():
    h = ["<h2>8. Bộ nhóm làm (sft_multiturn) so với raw BTC</h2>"]
    h.append("<p class=note>Trả lời câu hỏi: raw <code>train/</code> đã đủ để học, vậy có cần bộ nhóm làm không?</p>")
    h.append("<p><b>a) Raw đã có đủ.</b> <code>train.jsonl</code> có <code>context</code>, <code>history</code> (kèm <code>audio_text</code>, <code>tool_calls</code>, <code>tool_results</code>), <code>audio_text</code> của lượt cuối và <code>label</code> (<code>tool_calls</code>, <code>tool_results</code>, <code>final_state</code>). Catalog tool nằm ở file <code>train_tools_catalog.json</code>.</p>")
    h.append("<p><b>b) Bộ nhóm làm chỉ đổi dạng.</b> Nó ghép các trường trên (và schema tool theo <code>installed_apps</code>) thành <code>messages</code> + <code>tools</code> giống chat template. Nội dung giữ nguyên, chỉ bỏ 1 mẫu trùng <code>audio_text</code> (<code>train_0066</code>) và chia sẵn train 1244 / dev 300.</p>")
    h.append("<p><b>Tập dev để làm gì?</b> Dev là 300 mẫu (19,4% train) giữ lại để kiểm chứng, không dùng để học. Mục đích là đo xem agent có khái quát sang app lạ không, vì public_test có 11 app mới so với train.</p><ul>"
             "<li>228 mẫu gọi 9 app lạ chỉ có ở dev: <code>bike_rental</code>, <code>calculator</code>, <code>exchange_rate</code>, <code>fitness</code>, <code>fuel_price</code>, <code>gold_price</code>, <code>lottery</code>, <code>package_tracking</code>, <code>tv</code>.</li>"
             "<li>72 mẫu bổ sung để tỉ lệ nhãn bằng train: act 236, ask 42, empty 22.</li>"
             "<li>Dùng cho A/B, tối ưu prompt GEPA, baseline và kiểm cuối WSP. Sinh bằng <code>eval/splits/make_dev_split.py</code>, seed 42.</li></ul>")
    h.append("<p><b>c) Việc đổi dạng chỉ cần khi train bằng trainer.</b> Các thư viện SFT (LoRA, TRL...) thường đọc dạng <code>messages</code>, không đọc thẳng raw. Nếu tự viết bước đọc raw thì không cần file trung gian. Hiện pipeline và WSP không đọc bộ này.</p>")
    h.append("<h3>Một mẫu sau khi làm phẳng (train_0454, rút gọn)</h3>")
    h.append(f"<details open><summary>Xem mẫu</summary><pre style=\"overflow-x:auto;background:var(--panel);padding:10px 12px;border-radius:8px;font-size:12.5px;line-height:1.45\">{E(SFT_EXAMPLE)}</pre></details>")
    h.append("<h3>Trường cấp mẫu</h3>" + table(["trường", "ý nghĩa"], SFT_FIELDS_SAMPLE))
    h.append("<h3>Trường trong mỗi lượt của <code>messages</code></h3>" + table(["trường", "ý nghĩa"], SFT_FIELDS_MSG))
    h.append("<h3>Trường trong mỗi phần tử của <code>tools</code></h3>" + table(["trường", "ý nghĩa"], SFT_FIELDS_TOOL))
    return "".join(h)


def make_btc(date):
    cats = {sp: load_catalog(sp) for sp in ("train", "public_test")}
    version = (RAW / "VERSION").read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(l) for l in open(RAW / "train/train.jsonl", encoding="utf-8")]
    b = []
    b.append("<h2>1. Gói dữ liệu BTC có gì</h2>")
    b.append(f"<p class=note>Phiên bản (file VERSION): {E(' | '.join(version))}. Audio (.wav) không có trên máy này (PROGRESS A1).</p>")
    b.append(table(["thành phần", "train", "public_test", "ý nghĩa"], [
        ["`<split>.jsonl`", "1545 mẫu (có nhãn)", "634 mẫu (chỉ input, không nhãn)", "yêu cầu, context, history; train thêm audio_text và label"],
        ["`<split>_tools_catalog.json`", f"{len(cats['train'])} app, {sum(len(a['tools']) for a in cats['train'].values())} tool", f"{len(cats['public_test'])} app, {sum(len(a['tools']) for a in cats['public_test'].values())} tool", "schema tool: kind, tham số, enum, x-ref, x-one-of"],
        ["`<split>_environment/<id>/state.json`", "1545 thư mục", "634 thư mục (không đọc)", "dữ liệu thật của từng app tại thời điểm hỏi"],
        ["`zaic_tools/`", "có", "có", "thư viện chạy tool (ToolEnv) và tính state_hash"],
        ["`audio/`", "chưa có trên máy", "chưa có trên máy", "âm thanh yêu cầu"]], ()).replace("`", ""))
    b.append("<h2>2. Tổng quan catalog</h2><p class=note>Số app, tool, nhóm và loại tool. Chỉ đọc catalog, không đọc dữ liệu test.</p>")
    b.append(legend([(n, c) for n, c in KIND]))
    for sp, apps in cats.items():
        k = collections.Counter(kind for a in apps.values() for kind in a["kind"].values())
        b.append(f"<h3>{sp}</h3>" + kpis([("app", len(apps)), ("tool", sum(len(a['tools']) for a in apps.values())),
                                         ("nhóm app", len({a['group'] for a in apps.values()}))]) + formula(apps))
    b.append("<h2>3. Nhóm app: mỗi nhóm có bao nhiêu app, bao nhiêu tool</h2><p class=note>Nhãn bên trái ghi số app của nhóm; độ dài thanh là số tool, chia theo read/write/meta. Di chuột để xem tên các app trong nhóm.</p>")
    b.append(legend([(n, c) for n, c in KIND]))
    cols = []
    for sp, apps in cats.items():
        gr, _ = group_rows(apps)
        cols.append(card(sp, hbar(gr, ["read", "write", "meta"], label_w=150)))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    b.append("<h2>4. Tool theo app, gom theo nhóm</h2><p class=note>Mỗi thanh là một app; độ dài là số tool, chia theo read/write/meta.</p>")
    b.append(legend([(n, c) for n, c in KIND]))
    cols = []
    for sp, apps in cats.items():
        _, g = group_rows(apps)
        cols.append(card(sp, hbar(app_rows(apps, g), ["read", "write", "meta"], label_w=130)))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    tr, pt = cats["train"], cats["public_test"]
    both = sorted(set(tr) & set(pt)); only_tr = sorted(set(tr) - set(pt)); only_pt = sorted(set(pt) - set(tr))
    tt = {t for a in tr.values() for t in a["tools"]}; pp = {t for a in pt.values() for t in a["tools"]}
    t_both, t_tr, t_pt = sorted(tt & pp), sorted(tt - pp), sorted(pp - tt)
    for sp, apps in cats.items():
        groups = collections.defaultdict(list)
        for aid, a in apps.items():
            groups[a["group"]].append(aid)
        inner = []
        for gname in sorted(groups, key=lambda g: -sum(len(apps[x]["tools"]) for x in groups[g])):
            ids = sorted(groups[gname], key=lambda x: -len(apps[x]["tools"]))
            inner.append(f"<h3>Nhóm {E(gname)}: {len(ids)} app</h3>" + "".join(app_detail(x, apps[x]) for x in ids))
        b.append(f"<h2>4b. Danh mục app của {sp}</h2><p class=note>Mỗi app: tên, mô tả, và từng tool với loại (read/write/meta), mô tả và tham số bắt buộc. Bấm để mở.</p>" + "".join(inner))
    b.append("<h2>5. So sánh train và public_test</h2>")
    b.append("<p class=note>So theo tên app (<code>id</code>) và tên tool (<code>app.tool</code>). Một app được coi là trùng khi có id ở cả hai catalog; một tool trùng khi có cùng tên ở cả hai.</p>")
    b.append(kpis([("app trùng", len(both)), ("chỉ có ở train", len(only_tr)), ("chỉ có ở public_test", len(only_pt)),
                   ("tool trùng", len(t_both)), ("tool chỉ có ở train", len(t_tr)), ("tool chỉ có ở public_test", len(t_pt))]))
    same = [a for a in both if set(tr[a]["tools"]) == set(pt[a]["tools"])]
    b.append(f"<p class=note>Kết luận: {len(both)} app trùng, trong đó {len(same)} app có đúng cùng tập tool ở hai bên. {len(t_both)} tool trùng, {len(t_tr)} tool chỉ có ở train, {len(t_pt)} tool chỉ có ở public_test. Tổng train {len(tt)} = trùng {len(t_both)} + chỉ train {len(t_tr)}; tổng public_test {len(pp)} = trùng {len(t_both)} + chỉ public_test {len(t_pt)}.</p>")
    arows = []
    for a in both:
        same_ = set(tr[a]["tools"]) == set(pt[a]["tools"])
        arows.append([tr[a]["group"], a, tr[a]["name"], len(tr[a]["tools"]), len(pt[a]["tools"]), "giống hệt" if same_ else "khác"])
    b.append(f"<h3>App trùng ({len(both)} app)</h3>" + table(["nhóm", "app", "tên", "tool ở train", "tool ở public_test", "tập tool"], arows, (3, 4)))
    b.append(f"<details><summary>Tool trùng ({len(t_both)} tool): tên và loại</summary>" + table(["tool", "loại"], [[t, tr_kind(tr, t)] for t in t_both]) + "</details>")
    b.append(f"<h3>Chỉ có ở public_test ({len(only_pt)} app): train không có app này</h3>")
    b.append("".join(app_detail(a, pt[a]) for a in only_pt))
    b.append(f"<h3>Chỉ có ở train ({len(only_tr)} app): public_test không có app này</h3>")
    b.append("".join(app_detail(a, tr[a]) for a in only_tr))
    b.append(f"<p class=note>Ý nghĩa: public_test có {len(only_pt)} app và {len(t_pt)} tool mà train không có. Model chưa thấy các tool này khi học, nên phải dựa vào schema (mô tả tool và tham số) để gọi đúng.</p>")
    # mẫu train
    def cat(r):
        c = r["label"]["tool_calls"]
        return "ask" if any(x["name"] == "sys.ask_missing_info" for x in c) else ("empty" if not c else "act")
    cc = collections.Counter(cat(r) for r in rows)
    ncalls = collections.Counter(len(r["label"]["tool_calls"]) for r in rows)
    nh = collections.Counter(len(r["history"]) for r in rows)
    ninst = collections.Counter(len(r["context"]["installed_apps"]) for r in rows)
    b.append("<h2>6. Mẫu train của BTC (1545 mẫu)</h2><p class=note>Tính từ <code>train/train.jsonl</code>, trước khi nhóm tách dev và loại trùng.</p>")
    b.append(kpis([("mẫu", len(rows)), ("act", cc["act"]), ("ask", cc["ask"]), ("empty", cc["empty"])]))
    l1, v1 = hist(ncalls.elements()); l2, v2 = hist(nh.elements()); l3, v3 = hist(ninst.elements())
    b.append('<div class="cols">' + card("Số tool call gold mỗi mẫu", vbar(l1, v1)) + card("Số lượt trong history (user + assistant)", vbar(l2, v2)) + card("Số app cài mỗi mẫu", vbar(l3, v3)) + "</div>")
    b.append("<h2>7. Bảng chi tiết (view bằng bảng)</h2>")
    for sp, apps in cats.items():
        b.append(f"<details><summary>{sp}: {len(apps)} app</summary>{table(APP_HEAD, app_table_rows(apps), (3,4,5,6))}</details>")
    b.append(sft_vs_raw_section())
    return page("Thống kê dữ liệu BTC", f"Sinh ngày {date} bằng <code>scripts/make_stats_html.py</code>. Dữ liệu BTC, không đăng công khai.", "".join(b))


# ---------------------------------------------------------------- file 2: dữ liệu nhóm
def make_sft(date):
    cat = load_catalog("train")
    tool_kind = {t: k for a in cat.values() for t, k in a["kind"].items()}
    tool_desc = {t: ti["desc"] for a in cat.values() for t, ti in a["tool_info"].items()}
    tool_app = {t: aid for aid, a in cat.items() for t in a["tools"]}
    app_group = {aid: a["group"] for aid, a in cat.items()}
    data = {}
    for sp in ("train", "dev"):
        data[sp] = [json.loads(l) for l in open(ROOT / f"data/processed/sft_multiturn_{sp}_{date}.jsonl", encoding="utf-8")]
    def gold_calls(r):
        out = []
        for m in r["messages"]:
            if m["role"] == "assistant" and m.get("train"):
                out += [c["function"]["name"] for c in m["tool_calls"]]
        return out
    stat = {}
    for sp, rows in data.items():
        calls = collections.Counter(); app_calls = collections.Counter(); kind_by_app = collections.defaultdict(collections.Counter)
        kind_by_group = collections.defaultdict(collections.Counter)
        for r in rows:
            for c in gold_calls(r):
                calls[c] += 1; a = tool_app.get(c, "sys"); k = tool_kind.get(c, "meta")
                kind_by_app[a][k] += 1; kind_by_group[app_group.get(a, "control")][k] += 1
        stat[sp] = {"calls": calls, "kind_app": kind_by_app, "kind_group": kind_by_group}
    b = []
    allr = data["train"] + data["dev"]
    b.append("<h2>1. Tổng quan</h2>")
    b.append(formula(cat))
    b.append(kpis([("mẫu train", len(data["train"])), ("mẫu dev", len(data["dev"])),
                   ("app cài (hợp train+dev)", len({a for r in allr for a in r["apps"]})),
                   ("tool trong catalog train", len(tool_kind)),
                   ("token trung vị / mẫu", int(statistics.median(r["n_tokens"] for r in allr))),
                   ("mẫu > 8192 token", sum(r["n_tokens"] > 8192 for r in allr))]))
    b.append("<p class=note>\"App cài\" là app trong <code>installed_apps</code> của mẫu (mỗi mẫu cài khoảng 9 app). \"Tool call\" ở các mục sau chỉ tính call gold của lượt cuối (assistant có <code>train: true</code>).</p>")
    b.append("<h2>2. Loại mẫu (category)</h2>")
    cols = []
    for sp, rows in data.items():
        c = collections.Counter(r["category"] for r in rows)
        order = ["act", "ask", "empty"]
        cols.append(card(sp, vbar(order, [c[o] for o in order], tips=[f"{o}: {c[o]} ({c[o]/len(rows):.1%})" for o in order]),
                         "act = gọi tool; ask = gọi sys.ask_missing_info; empty = không gọi tool (ngoài khả năng)"))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    b.append("<h2>3. Tool call gold theo nhóm app, phân loại read / write / meta</h2>")
    b.append(legend([(n, c) for n, c in KIND]))
    cols = []
    for sp in data:
        rows = []
        for g, k in sorted(stat[sp]["kind_group"].items(), key=lambda kv: -sum(kv[1].values())):
            rows.append({"label": g, "vals": [k["read"], k["write"], k["meta"]], "tip": f"{g}: read {k['read']}, write {k['write']}, meta {k['meta']}"})
        tot = sum(sum(k.values()) for k in stat[sp]["kind_group"].values())
        kt = collections.Counter()
        for k in stat[sp]["kind_group"].values(): kt.update(k)
        cols.append(card(f"{sp}: {tot} call", hbar(rows, ["read", "write", "meta"], label_w=110), f"read {kt['read']}, write {kt['write']}, meta {kt['meta']}"))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    b.append("<h2>4. Tool call gold theo app, gom theo nhóm</h2>")
    b.append(legend([(n, c) for n, c in KIND]))
    cols = []
    for sp in data:
        rows = []
        gk = stat[sp]["kind_group"]
        for g, k in sorted(gk.items(), key=lambda kv: -sum(kv[1].values())):
            rows.append({"header": f"{g}: {sum(k.values())} call"})
            apps_g = [a for a in stat[sp]["kind_app"] if (app_group.get(a, "control")) == g]
            for a in sorted(apps_g, key=lambda x: -sum(stat[sp]["kind_app"][x].values())):
                kk = stat[sp]["kind_app"][a]
                rows.append({"label": a, "vals": [kk["read"], kk["write"], kk["meta"]], "tip": f"{a}: read {kk['read']}, write {kk['write']}, meta {kk['meta']}"})
        cols.append(card(sp, hbar(rows, ["read", "write", "meta"], label_w=130)))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    b.append("<h2>5. Độ phủ tool của catalog</h2>")
    cov = []
    for sp in data:
        used = set(stat[sp]["calls"]); allt = set(tool_kind)
        cov.append((sp, used, allt - used))
    b.append(kpis([(f"{sp}: tool được gọi / {len(tool_kind)}", len(u)) for sp, u, _ in cov]))
    tr_used, dv_used = cov[0][1], cov[1][1]
    b.append(f"<h3>Tool dev gọi nhưng train không gọi: {len(dv_used - tr_used)}</h3>" + ("<p class=note>Không có.</p>" if not (dv_used - tr_used) else "".join(f'<span class="tag">{E(t)}</span>' for t in sorted(dv_used - tr_used))))
    b.append(f"<h3>Tool train không gọi lần nào: {len(set(tool_kind) - tr_used)}</h3>" + "".join(f'<span class="tag" title="{E(tool_desc[t])}">{E(t)} ({tool_kind[t]})</span>' for t in sorted(set(tool_kind) - tr_used)))
    b.append("<h3>Top 20 tool được gọi nhiều nhất (train)</h3>")
    top = stat["train"]["calls"].most_common(20)
    rows = [{"label": t.split(".", 1)[1] if False else t, "vals": [n], "tip": f"{t} ({tool_kind.get(t,'meta')}): {n} call"} for t, n in top]
    b.append(card("train", hbar(rows, ["one"], label_w=200, width=520)))
    # dev app lạ
    apps_tr = {a for r in data["train"] for a in r["apps"]}; apps_dv = {a for r in data["dev"] for a in r["apps"]}
    called_tr = {t.split(".")[0] for t in stat["train"]["calls"]}; called_dv = {t.split(".")[0] for t in stat["dev"]["calls"]}
    lạ = sorted(called_dv - called_tr)
    b.append("<h2>6. Dev có app lạ so với train không?</h2>")
    b.append(kpis([("app cài trong train", len(apps_tr)), ("app cài trong dev", len(apps_dv)), ("app được gọi trong train", len(called_tr)),
                   ("app được gọi trong dev", len(called_dv)), ("app chỉ được gọi ở dev", len(lạ))]))
    b.append("<h3>App chỉ được gọi ở dev (train không có call nào)</h3>" + "".join(f'<span class="tag">{E(a)}</span>' for a in lạ))
    b.append("<p class=note>Cần phân biệt hai cách đếm. Theo app được cài (installed_apps) thì dev không có app nào ngoài train, vì train vẫn cài các app này nhưng không có nhãn gọi chúng. Theo app được gọi trong nhãn thì dev có 9 app lạ, đúng với <code>eval/splits/README.md</code>. Cách thứ hai mới là cách đo khả năng với app chưa thấy khi học.</p>")
    b.append("<h2>7. Phân bố độ dài</h2>")
    cols = []
    for title, fn, bins in (("Số tool call gold mỗi mẫu", lambda r: r["n_calls"], None), ("Số lượt history", lambda r: r["n_history"], None)):
        inner = []
        for sp, rows in data.items():
            l, v = hist([fn(r) for r in rows]); inner.append(card(f"{title} ({sp})", vbar(l, v)))
        cols += inner
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    cols = []
    for sp, rows in data.items():
        c = collections.Counter(min(r["n_tokens"] // 1000, 9) for r in rows)
        labels = [f"{i}k" if i < 9 else "9k+" for i in range(10)]
        cols.append(card(f"Token mỗi mẫu, bước 1000 ({sp})", vbar(labels, [c[i] for i in range(10)], tips=[f"{labels[i]}: {c[i]} mẫu" for i in range(10)]),
                         f"trung vị {int(statistics.median(r['n_tokens'] for r in rows))}, max {max(r['n_tokens'] for r in rows)} (tokenizer Qwen3.8-4B-Distilled, gồm tool schema)"))
    b.append(f'<div class="cols">{"".join(cols)}</div>')
    b.append("<h2>8. Bảng chi tiết (view bằng bảng)</h2>")
    for sp in data:
        trs = []
        for t, n in sorted(stat[sp]["calls"].items(), key=lambda kv: (-kv[1], kv[0])):
            trs.append([tool_app.get(t, "sys"), t, tool_kind.get(t, "meta"), n])
        b.append(f"<details><summary>{sp}: {len(trs)} tool được gọi, {sum(stat[sp]['calls'].values())} call</summary>{table(['app', 'tool', 'kind', 'số call'], trs, (3,))}</details>")
    return page("Thống kê bộ data nhóm (sft_multiturn)", f"Sinh ngày {date} bằng <code>scripts/make_stats_html.py</code> từ <code>data/processed/sft_multiturn_{{train,dev}}_{date}.jsonl</code>.", "".join(b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", default="20261111"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("stats_btc", make_btc), ("stats_sft_multiturn", make_sft)):
        p = OUT / f"{name}_{a.date}.html"
        p.write_text(fn(a.date), encoding="utf-8"); print("ghi", p, p.stat().st_size, "byte")


if __name__ == "__main__":
    main()
