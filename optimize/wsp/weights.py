"""Kernel đọc/ghi trọng số cho weight-space programming (WSP) trên Qwen3-4B.

Đọc tensor qua mmap (safetensors.safe_open), không nạp cả model vào RAM. Sửa được lưu trong bộ nhớ (`edits`)
cho tới khi gọi `save()`, nên có thể thử, đo, rồi bỏ.

Phép sửa chính theo blog Sparks of Weight-Space Programming (Thí nghiệm 8): trừ một hướng d (trong không gian
hidden) khỏi các cột đầu ra của MLP down_proj ở một tầng:
    W_down[:, cols] <- W_down[:, cols] - alpha * d (d^T W_down[:, cols])
Sau phép này, đóng góp của các neuron trong `cols` lên residual stream không còn thành phần theo hướng d.

Quy ước tên tensor (Qwen3): model.layers.{i}.mlp.{gate,up,down}_proj.weight; down_proj có shape [hidden, intermediate].
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class ModelWeights:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        idx = json.load(open(self.dir / "model.safetensors.index.json", encoding="utf-8"))
        self.map: dict[str, str] = idx["weight_map"]          # tên tensor -> tên shard
        self.config = json.load(open(self.dir / "config.json", encoding="utf-8"))
        self.edits: dict[str, torch.Tensor] = {}               # tên tensor -> tensor đã sửa
        self.log: list[dict] = []

    # ---------------------------------------------------------------- đọc
    @property
    def num_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    def names(self, contains: str = "") -> list[str]:
        return sorted(n for n in self.map if contains in n)

    def original(self, name: str) -> torch.Tensor:
        with safe_open(self.dir / self.map[name], framework="pt") as f:
            return f.get_tensor(name)

    def get(self, name: str) -> torch.Tensor:
        return self.edits[name] if name in self.edits else self.original(name)

    def shape(self, name: str) -> list[int]:
        with safe_open(self.dir / self.map[name], framework="pt") as f:
            return list(f.get_slice(name).get_shape())

    def total_params(self) -> int:
        total = 0
        for shard in sorted(set(self.map.values())):
            with safe_open(self.dir / shard, framework="pt") as f:
                for n in f.keys():
                    s = f.get_slice(n).get_shape()
                    p = 1
                    for x in s:
                        p *= x
                    total += p
        return total

    @staticmethod
    def mlp(layer: int, kind: str = "down") -> str:
        return f"model.layers.{layer}.mlp.{kind}_proj.weight"

    # ---------------------------------------------------------------- sửa
    def top_cols(self, name: str, direction: torch.Tensor, k: int) -> list[int]:
        """k cột của down_proj ghi mạnh nhất theo hướng d: |d^T W[:, j]| lớn nhất."""
        W = self.get(name).float()
        d = direction.float().flatten()
        d = d / d.norm()
        score = (d @ W).abs()
        return torch.topk(score, k).indices.sort().values.tolist()

    def project_out(self, name: str, direction: torch.Tensor, cols: list[int] | None = None,
                    alpha: float = 1.0, note: str = "") -> int:
        """Trừ hướng d khỏi các cột `cols` (None = mọi cột). Trả về số phần tử thực sự đổi."""
        W = self.get(name)
        d = direction.float().flatten()
        if W.shape[0] != d.numel():
            raise ValueError(f"{name}: hidden {W.shape[0]} khác độ dài hướng {d.numel()}")
        d = d / d.norm()
        Wf = W.float().clone()
        idx = torch.arange(W.shape[1]) if cols is None else torch.as_tensor(cols, dtype=torch.long)
        sub = Wf[:, idx]
        Wf[:, idx] = sub - alpha * torch.outer(d, d @ sub)
        new = Wf.to(W.dtype)
        self.edits[name] = new
        changed = int((new != self.original(name)).sum())
        self.log.append({"op": "project_out", "tensor": name, "n_cols": int(idx.numel()), "alpha": alpha,
                         "changed_vs_original": changed, "note": note})
        return changed

    def scale_cols(self, name: str, cols: list[int], factor: float, note: str = "") -> int:
        W = self.get(name)
        Wf = W.float().clone()
        idx = torch.as_tensor(cols, dtype=torch.long)
        Wf[:, idx] *= factor
        new = Wf.to(W.dtype)
        self.edits[name] = new
        changed = int((new != self.original(name)).sum())
        self.log.append({"op": "scale_cols", "tensor": name, "n_cols": int(idx.numel()), "factor": factor,
                         "changed_vs_original": changed, "note": note})
        return changed

    def revert(self, name: str | None = None) -> None:
        if name is None:
            self.edits.clear()
            self.log.append({"op": "revert_all"})
        else:
            self.edits.pop(name, None)
            self.log.append({"op": "revert", "tensor": name})

    def n_changed(self) -> int:
        return sum(int((t != self.original(n)).sum()) for n, t in self.edits.items())

    # ---------------------------------------------------------------- lưu
    def save(self, out_dir: str | Path) -> dict:
        """Ghi model đầy đủ (vLLM đọc được) vào out_dir. Shard không bị sửa thì hardlink (cùng ổ) hoặc copy."""
        out = Path(out_dir)
        if out.resolve() == self.dir.resolve():
            raise ValueError("không ghi đè model gốc")
        out.mkdir(parents=True, exist_ok=True)
        for p in self.dir.iterdir():                           # config, tokenizer, index, ...
            if p.is_file() and not p.name.endswith(".safetensors"):
                shutil.copy2(p, out / p.name)
        by_shard: dict[str, list[str]] = {}
        for n in self.edits:
            by_shard.setdefault(self.map[n], []).append(n)
        for shard in sorted(set(self.map.values())):
            src, dst = self.dir / shard, out / shard
            if dst.exists():
                dst.unlink()
            if shard in by_shard:
                tensors = load_file(src)
                for n in by_shard[shard]:
                    tensors[n] = self.edits[n].contiguous()
                save_file(tensors, dst, metadata={"format": "pt"})
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        total = self.total_params()
        changed = self.n_changed()
        stats = {"base_model": str(self.dir), "edited_tensors": sorted(self.edits), "n_changed": changed,
                 "total_params": total, "fraction_changed": changed / total, "log": self.log}
        json.dump(stats, open(out / "wsp_edits.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return stats


def apply_spec(mw: ModelWeights, spec: list[dict], directions: dict[int, torch.Tensor] | None = None) -> int:
    """Áp danh sách phép sửa dạng JSON. Mỗi phép:
    {"op": "project_out", "layer": 18, "mat": "down", "direction_layer": 18, "top_k_cols": 64, "alpha": 1.0}
    `direction_layer` lấy hướng từ `directions` (đầu ra directions.py). "cols" thay "top_k_cols" nếu muốn chỉ định cột."""
    for op in spec:
        name = mw.mlp(op["layer"], op.get("mat", "down"))
        if op["op"] == "project_out":
            if directions is None:
                raise ValueError("project_out cần directions")
            d = directions[int(op.get("direction_layer", op["layer"]))]
            cols = op.get("cols") or (mw.top_cols(name, d, int(op["top_k_cols"])) if op.get("top_k_cols") else None)
            mw.project_out(name, d, cols, float(op.get("alpha", 1.0)), note=op.get("note", ""))
        elif op["op"] == "scale_cols":
            mw.scale_cols(name, op["cols"], float(op["factor"]), note=op.get("note", ""))
        else:
            raise ValueError(f"phép không hỗ trợ: {op['op']}")
    return mw.n_changed()
