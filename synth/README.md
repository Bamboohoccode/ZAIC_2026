# Synthetic data [Kiệt]

Bộ 3 nhãn sinh từ train (có nhãn sẵn, không gán nhãn tay, không đụng test): `full`, `missing_slot`, `no_app`.

```
python synth/gen_three_labels.py --root data          # -> data/synth/train_synth.jsonl (data/ không commit)
```

- `full`: mẫu act gốc.
- `no_app`: gỡ app đáp án dùng (và app cùng `group` trong catalog) khỏi `installed_apps`, đáp án `[]`.
- `missing_slot`: xoá giá trị một tham số bắt buộc của call ghi cuối khỏi câu nói, đáp án = call đọc trước đó + `sys.ask_missing_info`.

Mỗi mẫu giữ `id` gốc (ToolEnv dùng đúng môi trường) và thêm `synth`, `synth_id`, `synth_app`; `audio` = None nên chỉ chạy `--text`.
GEPA của Kiệt dùng `missing_slot` và `no_app` làm thêm dữ liệu train (`synth_path` trong config), mẫu có nguồn thuộc dev chỉ để chấm.
