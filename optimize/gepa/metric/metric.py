"""Metric GEPA dùng chung (Kiệt, Vương).

evaluate_sample(G, calls, state_hash, gold_state_hash, catalog, hash_credit)   # Planner, điểm cuối theo BTC
    -> {score, shaped, kind, error, feedback}
    score:  điểm BTC (eval/scorer.py), dùng để báo cáo.
    shaped: điểm đưa cho GEPA. Giống score, nhưng mẫu lệch hash được hash_credit * F1 (F1 bỏ qua hash) thay vì 0,
            vì baseline có 46% mẫu lệch hash nên điểm 0/1 gần như không phân biệt được hai prompt trong một minibatch.
            hash_credit = 0 thì shaped == score. Báo cáo cuối luôn dùng score.

evaluate_mapper(raw, mapped, gold_text, G)   # Mapper, điểm nhanh không cần chạy Planner
    -> {score, wer_raw, wer, entities, feedback}
    score = 0.7 * tỉ lệ giữ đúng thực thể + 0.3 * (1 - WER), thực thể = giá trị chuỗi trong tool call đáp án có mặt
    trong audio_text (tên, địa điểm...). Không có thực thể thì score = 1 - WER. Dùng để tối ưu cho rẻ GPU;
    chốt Mapper vẫn bằng điểm cuối (run_gepa eval --mapper ...).
"""
import re
import unicodedata

from eval.scorer import match_calls, score_sample, task_kind
from optimize.gepa.metric.feedback import explain, text_diff

ENT_WEIGHT = 0.7


def evaluate_sample(G, calls, state_hash, gold_state_hash, catalog, hash_credit: float = 0.0) -> dict:
    score, err = score_sample(G, calls, state_hash, gold_state_hash, catalog)
    pairs, extra, missing = match_calls(calls, G, catalog)
    shaped = score
    if err == "HASH_MISMATCH":
        shaped = hash_credit * 2 * len(pairs) / (len(calls) + len(G))
    return {"score": score, "shaped": shaped, "kind": task_kind(G), "error": err,
            "feedback": explain(G, calls, err, pairs, extra, missing)}


def words(s):
    return re.findall(r"\w+", unicodedata.normalize("NFC", str(s)).lower())


def wer(hyp, ref):
    """Word error rate (Levenshtein theo từ) / số từ của ref."""
    h, r = words(hyp), words(ref)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def entities(G, gold_text):
    """Giá trị chuỗi (>= 2 ký tự) trong tool call đáp án xuất hiện nguyên cụm trong audio_text: Mapper phải giữ hoặc sửa ra đúng."""
    text = f" {' '.join(words(gold_text))} "
    vals = [v for g in G for v in (g.get("arguments") or {}).values() if isinstance(v, str) and len(v.strip()) >= 2]
    return list(dict.fromkeys(" ".join(words(v)) for v in vals if words(v) and f" {' '.join(words(v))} " in text))


def evaluate_mapper(raw, mapped, gold_text, G) -> dict:
    ents, text = entities(G, gold_text), f" {' '.join(words(mapped))} "
    kept = [e for e in ents if f" {e} " in text]
    w, w_raw = wer(mapped, gold_text), wer(raw, gold_text)
    score = (1 - min(1.0, w)) if not ents else ENT_WEIGHT * len(kept) / len(ents) + (1 - ENT_WEIGHT) * (1 - min(1.0, w))
    if score >= 0.999:
        fb = "Đúng."
    else:
        fb = text_diff(raw, mapped, gold_text) + f"\nWER: ASR {w_raw:.2f} -> Mapper {w:.2f}"
        if ents:
            fb += f"\nThực thể Planner cần: {', '.join(ents)}; sai hoặc mất: {', '.join(e for e in ents if e not in kept) or 'không'}"
        if w > w_raw:
            fb += "\nMapper làm câu tệ hơn bản ASR: chỉ sửa khi chắc chắn, còn lại chép y nguyên."
    return {"score": score, "wer_raw": w_raw, "wer": w, "entities": ents, "feedback": fb}
