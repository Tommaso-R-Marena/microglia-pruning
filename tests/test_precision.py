import torch

from src.precision import MixedPrecisionTrainer, PrecisionConfig


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def test_precision_config_dtype_mapping():
    assert PrecisionConfig("fp16").dtype == torch.float16
    assert PrecisionConfig("bf16").dtype == torch.bfloat16
    assert PrecisionConfig("fp32").dtype == torch.float32


def test_train_step_updates_parameters():
    model = TinyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = MixedPrecisionTrainer(model, optim, PrecisionConfig("fp32"))

    x = torch.randn(2, 4)
    y = torch.randn(2, 1)

    before = model.linear.weight.detach().clone()

    def loss_fn(batch_x, batch_y):
        pred = model(batch_x)
        return torch.nn.functional.mse_loss(pred, batch_y)

    loss = trainer.train_step(loss_fn, x, y)
    assert loss >= 0
    assert not torch.equal(before, model.linear.weight.detach())
