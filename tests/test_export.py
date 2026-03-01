from pathlib import Path

import torch

from src.export import export_to_onnx


class DummyLM(torch.nn.Module):
    def __init__(self, vocab_size=1000, hidden=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.head(x)
        return logits


def test_export_to_onnx_creates_file(tmp_path, monkeypatch):
    model = DummyLM()
    out = tmp_path / "dummy.onnx"

    def fake_export(model, inputs, path, **kwargs):
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    path = export_to_onnx(model, str(out), seq_len=4, vocab_size=32)
    assert path == out
    assert Path(path).exists()
