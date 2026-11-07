"""python -m tests.test_gepa_metric    # scorer, metric/feedback GEPA, chia dữ liệu và ghép prompt; không cần GPU hay data/"""
import random
import sys
import json
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.scorer import evaluate_submission, score_sample, task_app, task_kind
from eval.run_eval import pass_k
from optimize.gepa.kiet.run_gepa import join_prompt, pick, split_data, split_prompt
from optimize.gepa.metric.metric import evaluate_sample

CAT = {"a.get": {"parameters": {"properties": {"q": {"x-match": "fuzzy"}}}},
       "a.set": {"parameters": {"properties": {"v": {}, "name": {"x-match": "fuzzy"}}}}}
GET = {"name": "a.get", "arguments": {"q": "giá vàng sjc"}}
SET = {"name": "a.set", "arguments": {"v": 1, "name": "lan nguyễn"}}
ASK = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "a.set", "params": ["v"]}}


def test_score_sample():
    s = lambda G, c, h="H": score_sample(G, c, h, "H", CAT)
    assert s([], []) == (1.0, None) and s([], [GET]) == (0.0, "FALSE_ACT")
    assert s([], [], "X") == (0.0, "HASH_MISMATCH")  # hash vẫn phải đúng với gold rỗng
    assert s([GET, SET], [SET, GET]) == (1.0, None)  # không xét thứ tự
    assert s([GET], [{"name": "a.get", "arguments": {"q": "giá vàng sjc hôm nay"}}]) == (1.0, None)  # fuzzy token-F1 > 0.6
    assert s([GET, SET], [GET, SET], "X") == (0.0, "HASH_MISMATCH")
    assert s([GET], [GET, GET]) == (2 / 3, "EXTRA_CALL") and s([GET, SET], [GET]) == (2 / 3, "MISSING_CALL")
    assert s([GET], [GET, ASK]) == (2 / 3, "FALSE_ASK") and s([GET, ASK], [GET, SET]) == (0.5, "MISSED_ASK")
    assert s([SET], [{"name": "a.set", "arguments": {"v": 2, "name": "lan nguyễn"}}]) == (0.0, "WRONG_TOOL_OR_ARGS")
    assert (task_kind([]), task_kind([GET]), task_kind([GET, ASK])) == ("empty", "act", "ask")
    assert (task_app([]), task_app([GET]), task_app([ASK])) == ("(none)", "a", "a")


def test_pass_k_missing_id():
    score, domain = pass_k([{"scores": {"a": 1.0, "b": 1.0}}, {"scores": {"a": 1.0}}])
    assert domain == {"a", "b"} and score == 0.5


def test_metric_feedback():
    ok = evaluate_sample([GET], [GET], "H", "H", CAT)
    assert ok["score"] == 1.0 and ok["feedback"] == "Đúng." and ok["error"] is None
    m = evaluate_sample([GET, SET], [GET, {"name": "a.set", "arguments": {"v": 2, "name": "lan nguyễn"}}], "X", "H", CAT, hash_credit=0.2)
    assert m["score"] == 0.0 and abs(m["shaped"] - 0.2 * 0.5) < 1e-9 and m["error"] == "HASH_MISMATCH"
    assert "a.set: v: gọi 2, đúng 1" in m["feedback"] and "Đáp án:" in m["feedback"]
    assert evaluate_sample([GET, SET], [GET, SET], "X", "H", CAT)["shaped"] == 0.0  # hash_credit mặc định 0 = đúng BTC
    f = evaluate_sample([GET, ASK], [GET, SET], "H", "H", CAT)["feedback"]
    assert "sys.ask_missing_info" in f and "Thiếu:" in f and "Thừa:" in f
    assert "không gọi tool nào" in evaluate_sample([], [GET], "H", "H", CAT)["feedback"]


def test_prompt_roundtrip():
    tpl = "Giờ {now}.\nLịch:\n{dates}\n\nGọi <tool_call>{{\"name\": 1}}</tool_call> nhé."
    head, body = split_prompt(tpl)
    assert head.endswith("{dates}\n") and body == '\nGọi <tool_call>{"name": 1}</tool_call> nhé.'
    assert join_prompt(head, body) == tpl
    new = join_prompt(head, 'Ví dụ {"a": {"b": 2}}')  # LLM viết JSON ví dụ: format() không được vỡ
    assert new.format(now="n", dates="d") == 'Giờ n.\nLịch:\nd\nVí dụ {"a": {"b": 2}}'


def _item(i, app, kind):
    G = [] if kind == "empty" else [{"name": f"{app}.get", "arguments": {}}] + ([ASK] if kind == "ask" else [])
    return {"id": f"s{i}", "label": {"tool_calls": G}}


def test_split_data():
    rng = random.Random(0)
    items = [_item(i, f"app{i % 10}", rng.choices(["act", "ask", "empty"], [6, 3, 1])[0]) for i in range(600)]
    dev = {i["id"] for i in items[:100]}
    cfg = {"seed": 1, "val_app_frac": 0.2, "n_train": 100, "n_val": 40, "mix": {"act": 0.5, "ask": 0.3, "empty": 0.2}}
    tr, va = split_data(items, dev, cfg)
    ids = lambda xs: {i["id"] for i in xs}
    assert not (ids(tr) | ids(va)) & dev and not ids(tr) & ids(va)  # dev không rò vào tối ưu
    apps = lambda xs: {task_app(i["label"]["tool_calls"]) for i in xs} - {"(none)"}
    assert not apps(tr) & apps(va) and len(apps(va)) == 2  # val là app lạ
    assert (tr, va) == split_data(items, dev, cfg)  # tái lập
    k = [task_kind(i["label"]["tool_calls"]) for i in pick(items, 100, cfg["mix"], rng)]
    assert (k.count("act"), k.count("ask"), k.count("empty")) == (50, 30, 20)


def test_synth():
    from synth.gen_three_labels import generate, missing_slot, no_app
    cat = {"ewallet.transfer": {"kind": "write", "parameters": {"required": ["contact", "amount"]}},
           "contacts.search": {"kind": "read", "parameters": {"required": ["query"]}}}
    groups = {"ewallet": "finance", "bank": "finance", "maps": "nav"}
    T = {"name": "ewallet.transfer", "arguments": {"contact": "Lan", "amount": 200000}}
    it = {"id": "t1", "audio": "a.wav", "audio_text": "chuyển cho lan hai trăm nghìn nha", "history": [],
          "context": {"installed_apps": {"ewallet": {}, "bank": {}, "maps": {}}}, "label": {"tool_calls": [T]}}
    m = missing_slot(it, cat)
    assert m["audio_text"] == "chuyển cho hai trăm nghìn nha" and m["audio"] is None and m["id"] == "t1"
    assert m["label"]["tool_calls"] == [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["contact"]}}]
    n = no_app(it, groups)
    assert list(n["context"]["installed_apps"]) == ["maps"] and n["label"]["tool_calls"] == []  # gỡ cả app cùng group
    assert it["label"]["tool_calls"] == [T] and it["audio"] == "a.wav"  # không sửa mẫu gốc
    assert missing_slot({**it, "history": [{"role": "user", "audio_text": "lan là bạn tôi"}]}, cat) is None  # lịch sử đã có giá trị
    S = {"name": "contacts.search", "arguments": {"query": "Lan"}}
    assert missing_slot({**it, "label": {"tool_calls": [S, T]}}, cat) is None  # call đọc trước đã dùng giá trị
    assert missing_slot({**it, "audio_text": "chuyển cho lan lan"}, cat) is None  # giá trị xuất hiện 2 lần
    generated = generate([it], cat, groups)
    assert [s["synth"] for s in generated] == ["full", "no_app", "missing_slot"]
    assert all(s["audio"] is None for s in generated)  # synth chỉ chạy theo text


def test_eval_missing_submission():
    # Tạo split tối thiểu và ToolEnv giả; không truy cập state hay data BTC.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        split_dir = root / "train"
        split_dir.mkdir()
        (split_dir / "train_tools_catalog.json").write_text(json.dumps({"apps": []}), encoding="utf-8")
        (split_dir / "train.jsonl").write_text(
            json.dumps({"id": "sample-1", "label": {"tool_calls": []}}) + "\n", encoding="utf-8")
        empty_sub = root / "empty.jsonl"
        empty_sub.write_text("", encoding="utf-8")

        fake_tools = types.ModuleType("zaic_tools")
        fake_tools.ToolEnv = type("ToolEnv", (), {})
        previous = sys.modules.get("zaic_tools")
        sys.modules["zaic_tools"] = fake_tools
        try:
            result = evaluate_submission(
                str(root), "train", str(empty_sub), filter_ids={"sample-1"}, verbose=False)
        finally:
            if str(split_dir) in sys.path:
                sys.path.remove(str(split_dir))
            if previous is None:
                sys.modules.pop("zaic_tools", None)
            else:
                sys.modules["zaic_tools"] = previous

        assert result["total_samples"] == 1 and result["mean_score"] == 0.0
        assert result["diagnostics"][0]["error"] == "MISSING_SUBMISSION"


def test_mapper():
    from optimize.gepa.kiet.run_gepa import clean_mapped, map_turns, mapper_pool
    from optimize.gepa.metric.feedback import text_diff
    assert clean_mapped('Câu đã sửa: "bật đèn phòng khách"\nGiải thích...', "x") == "bật đèn phòng khách"
    assert clean_mapped("  \n", "giữ asr") == "giữ asr"

    class Echo:  # LLM giả: trả lại dòng giả thuyết đầu, ghi lại input
        seen = []

        def chat(self, convs, tools=None):
            self.seen += convs
            return [c[1]["content"].split("\n")[1].upper() for c in convs]
    it = {"id": "t1", "context": {"installed_apps": {"maps": {}}}, "audio_text": "x"}
    asr = {"t1": [["chào"], ["bật đèn", "bặt đèn"]]}
    llm = Echo()
    vocab = lambda i: "App đã cài: maps (Bản đồ)"
    assert map_turns(llm, "P", [it], asr, vocab) == [["CHÀO", "1. BẬT ĐÈN"]]  # mỗi lượt user một câu, n-best đánh số
    assert llm.seen[0][0] == {"role": "system", "content": "P"} and "App đã cài: maps (Bản đồ)" in llm.seen[1][1]["content"]
    assert map_turns(llm, "P", [it], asr, vocab, last_only=True) == [["1. BẬT ĐÈN"]]
    f = text_diff("bậc đèn phòng khác", "bật đèn phòng khác", "bật đèn phòng khách")
    assert "Mapper sửa đúng: bật" in f and "Còn sai chưa sửa: khách" in f and "Mapper làm mất từ vốn đúng: không" in f
    items = [{"id": f"s{i}", "audio_text": "a b"} for i in range(8)]
    asr2 = {f"s{i}": [["a c" if i < 4 else "a b"]] for i in range(8)}
    pool = mapper_pool(items, asr2, {"seed": 0, "mapper_same_frac": 0.25})
    assert [i["id"] for i in pool[:4]] == ["s0", "s1", "s2", "s3"] and len(pool) == 5  # 4 mẫu ASR sai + 1 mẫu đúng


def test_mapper_metric():
    from optimize.gepa.metric.metric import evaluate_mapper, wer
    assert wer("a b c", "a b c") == 0 and wer("a x c", "a b c") == 1 / 3 and wer("", "a b") == 1.0
    G = [{"name": "ewallet.transfer", "arguments": {"contact": "Lan Anh", "note": "tiền ăn", "amount": 5}}]
    gold = "chuyển cho lan anh năm nghìn"
    ok = evaluate_mapper("chuyển cho lan an năm nghìn", gold, gold, G)
    assert ok["score"] == 1.0 and ok["entities"] == ["lan anh"] and ok["feedback"] == "Đúng."  # "tiền ăn" không có trong câu
    bad = evaluate_mapper("chuyển cho lan an năm nghìn", "chuyển cho lan năm nghìn", gold, G)
    assert abs(bad["score"] - 0.3 * (1 - 1 / 6)) < 1e-9 and "sai hoặc mất: lan anh" in bad["feedback"]
    worse = evaluate_mapper(gold, "chuyển lan anh", gold, G)
    assert "tệ hơn bản ASR" in worse["feedback"]
    assert evaluate_mapper("bật đèn", "bật đèn", "bật đèn", [])["score"] == 1.0  # không có thực thể: chỉ WER


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
