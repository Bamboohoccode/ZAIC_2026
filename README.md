# ZAIC_2026

Repo dự án **LifeMate** cho Zalo AI Challenge 2026: agent nhận giọng nói tiếng Việt và gọi đúng chuỗi tool của các app mô phỏng.

## Tài liệu

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md): tổng quan dự án, pipeline, phân công, cấu trúc repo, rủi ro.
- [docs/DECISIONS.md](docs/DECISIONS.md): nhật ký quyết định và kết quả ablation.

## Cấu trúc

```
ZAIC_2026/
├── README.md
├── .gitignore
├── docs/                   # tổng quan, plan, luật, quyết định
├── configs/                # budget.yaml, pipeline.yaml
├── data/                   # KHÔNG commit (dữ liệu BTC)
├── lifemate/               # code module: asr, mapper, state, planner, validator, verifier, executor
│   ├── pipeline.py         # ghép pipeline (bản chạy được, chuyển từ gốc)
│   └── mapper/train.py     # LoRA fine-tune mapper phương ngữ
├── prompts/                # prompt đã chốt, kết quả GEPA
├── optimize/               # gepa, wsp
├── eval/                   # scorer, run_eval, splits
├── synth/                  # sinh dữ liệu 3 nhãn
├── docker/                 # Dockerfile, suy luận offline
├── scripts/                # count_params.py, bench_latency.py
├── tests/
│   └── test_pipeline.py    # unit test và chấm điểm cục bộ
├── ITN/                    # submodule rỗng, pipeline.py gọi khi chạy --itn
└── test_pipeline.ipynb     # notebook thử nghiệm
```

Chi tiết từng thư mục và người phụ trách xem [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) mục 3.

## Quy tắc nhanh

- Không commit `data/` hay trọng số model (luật cuộc thi cấm phát tán dữ liệu).
- Nhánh: `<loại>/<module>-<tên>`, ví dụ `feat/asr-khanh`. Chi tiết trong [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) mục 3.1.
- Chạy offline hoàn toàn khi suy luận.
