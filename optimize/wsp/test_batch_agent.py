"""Kiểm batch_agent.run_items cho cùng kết quả với lifemate/pipeline.main (không cần GPU).

Dùng một LLM giả, đầu ra là hàm thuần của hội thoại: planner phát lại gold từng call một (đôi khi bịa ID hoặc sai định dạng),
verifier REJECT theo hash nội dung. pipeline.main chạy thật (VLLM bị thay bằng LLM giả), batch_agent chạy với engine giả.
So submission từng byte, trace bỏ trường "seconds", "gen_tokens", "llm_calls".

Chạy: python optimize/wsp/test_batch_agent.py --root data/raw/train+public_test --n 60
"""
import argparse, hashlib, json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from lifemate import pipeline as P  # noqa: E402
import batch_agent as B  # noqa: E402


def h(s):
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class Fake:
    gold = {}  # audio_text -> list call gold

    @classmethod
    def reply(cls, msgs, tools):
        last = msgs[-1]["content"]
        if msgs[-1]["role"] == "user" and last.startswith("[KIỂM TRA]"):
            return '{"verdict": "REJECT", "reason": "sửa lại"}' if h(last) % 3 == 0 else '{"verdict": "ACCEPT"}'
        ui = max(i for i, m in enumerate(msgs) if m["role"] == "user" and not m["content"].startswith("["))
        q = msgs[ui]["content"]
        n = sum(1 for m in msgs[ui:] if m["role"] == "assistant" and m.get("tool_calls"))
        calls = cls.gold.get(q, [])
        r = h(q + str(n) + str(len(msgs)))
        if n == 1 and h(q) % 9 == 0:  # lỗi giữa chừng: buộc nhánh except (giữ call đã chạy)
            raise ValueError("lỗi giả của LLM")
        if n >= len(calls) or r % 11 == 0:
            return "<think>xong</think>Đã xong."
        c = json.loads(json.dumps(calls[n]))
        if r % 7 == 0:  # bịa ID: buộc nhánh unseen_ids
            for k, v in c["arguments"].items():
                if isinstance(v, str):
                    c["arguments"][k] = "fake_id_123"
                    break
        if r % 13 == 0:
            return "<think>a</think><tool_call>{không phải json}</tool_call>"
        return f"<think>bước {n}</think>\n<tool_call>{json.dumps(c, ensure_ascii=False)}</tool_call>"


class FakeSeq:  # thay P.VLLM trong pipeline.main
    def __init__(self, *a, **k):
        pass

    def n_tokens(self, msgs, tools):
        return 0

    def chat(self, convs, tools=None):
        return [Fake.reply(c, tools) for c in convs]


class FakeBatch:
    def n_tokens(self, msgs, tools):
        return 0

    def run(self, reqs):
        out = []
        for m, t in reqs:
            try:
                out.append(Fake.reply(m, t))
            except Exception as e:  # noqa: BLE001 - như VLLMBatch: lỗi của một mẫu trả về dạng Exception
                out.append(e)
        return out, [1] * len(reqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data" / "raw" / "train+public_test"))
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    root = Path(a.root)
    items = [json.loads(l) for l in open(root / "train" / "train.jsonl", encoding="utf-8") if l.strip()][: a.n]
    Fake.gold = {it["audio_text"]: it["label"]["tool_calls"] for it in items}
    tmp = Path(tempfile.mkdtemp())
    ids = tmp / "ids.txt"
    ids.write_text("\n".join(it["id"] for it in items), encoding="utf-8")

    P.VLLM = FakeSeq
    seq_out = tmp / "seq.jsonl"
    sys.argv = ["pipeline", "--root", str(root), "--split", "train", "--text", "--mapper", "none", "--llm", "fake",
                "--ids", str(ids), "--out", str(seq_out), "--seed", "42"]
    P.main()

    sys.path.insert(0, str(root / "train"))
    catalog = {t["name"]: t for app in json.load(open(root / "train" / "train_tools_catalog.json", encoding="utf-8"))["apps"]
               for t in app["tools"]}
    bat_out = tmp / "bat.jsonl"
    B.run_items(FakeBatch(), root, "train", B.load_items(root, "train", ids), catalog, P.VERIFY_ROUNDS, bat_out, log_every=0)

    s1, s2 = seq_out.read_text(encoding="utf-8"), bat_out.read_text(encoding="utf-8")
    drop = ("seconds", "gen_tokens", "llm_calls")
    t1 = [{k: v for k, v in json.loads(l).items() if k not in drop} for l in open(str(seq_out) + ".trace.jsonl", encoding="utf-8")]
    t2 = [{k: v for k, v in json.loads(l).items() if k not in drop} for l in open(str(bat_out) + ".trace.jsonl", encoding="utf-8")]
    n_verify = sum(1 for t in t1 for s in t["steps"] if "verify" in s)
    n_reject = sum(1 for t in t1 for s in t["steps"] if "verify" in s and "REJECT" in s["verify"])
    n_err = sum(1 for t in t1 for s in t["steps"] if "error" in s)
    print(f"mẫu {len(t1)} | verify {n_verify} (REJECT {n_reject}) | lỗi {n_err}")
    print("submission giống nhau:", s1 == s2)
    print("trace giống nhau:", t1 == t2)
    if s1 != s2 or t1 != t2:
        for x, y in zip(t1, t2):
            if x != y:
                print("KHÁC ở", x["id"])
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
