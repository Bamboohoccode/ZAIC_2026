import json

dev_ids = set(open('eval/splits/dev_ids.txt').read().split())
gold_dict = {item['id']: item for l in open('data/train/train.jsonl', encoding='utf-8') if (item := json.loads(l))['id'] in dev_ids}
pred_dict = {item['id']: item for l in open('optimize/gepa/vuong/eval_results/dev_v4_guarded.jsonl', encoding='utf-8') if (item := json.loads(l))['id'] in dev_ids}

print('=== 5 MẪU ĐIỂN HÌNH: THIẾU_TOOL_GHI_TRONG_CHUỖI ===')
count = 0
for sid, g in gold_dict.items():
    p = pred_dict.get(sid, {})
    g_calls = g.get('label', {}).get('tool_calls', [])
    p_calls = p.get('tool_calls', [])
    g_names = [c['name'] for c in g_calls]
    p_names = [c['name'] for c in p_calls]
    if len(p_names) < len(g_names) and count < 5:
        count += 1
        print(f"\n[{count}] ID: {sid}")
        print(f"Query: \"{g['audio_text']}\"")
        print(f"Gold calls ({len(g_calls)}): {g_names}")
        print(f"Pred calls ({len(p_calls)}): {p_names}")
        print("Diff calls:")
        print("  Gold:", g_calls)
        print("  Pred:", p_calls)

print('\n' + '='*70)
print('=== 5 MẪU ĐIỂN HÌNH: LỆCH_GIÁ_TRỊ_THAM_SỐ (CÙNG TOOL) ===')
count2 = 0
for sid, g in gold_dict.items():
    p = pred_dict.get(sid, {})
    g_calls = g.get('label', {}).get('tool_calls', [])
    p_calls = p.get('tool_calls', [])
    g_names = [c['name'] for c in g_calls]
    p_names = [c['name'] for c in p_calls]
    if g_names == p_names and g_names and count2 < 5:
        # check if args differ
        diff_args = []
        for gc, pc in zip(g_calls, p_calls):
            for k in set(gc.get('arguments', {}).keys()) | set(pc.get('arguments', {}).keys()):
                if gc.get('arguments', {}).get(k) != pc.get('arguments', {}).get(k):
                    diff_args.append((k, gc.get('arguments', {}).get(k), pc.get('arguments', {}).get(k)))
        if diff_args:
            count2 += 1
            print(f"\n[{count2}] ID: {sid}")
            print(f"Query: \"{g['audio_text']}\"")
            print(f"Tool: {g_names}")
            for k, gv, pv in diff_args:
                print(f"  -> Tham số '{k}': Gold={gv} vs Pred={pv}")
