import lightning.pytorch as pl
import torch

from classification.lit_data_module import LitGoogleFonts
from classification.lit_module import LitFontClassifier

torch.set_float32_matmul_precision("medium")


def main() -> None:
    data_module = LitGoogleFonts(max_len=256)
    model = LitFontClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
