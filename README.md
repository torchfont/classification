# Classification

[TorchFont](https://github.com/torchfont/torchfont) を使ったフォント文字分類のサンプルプロジェクトです。
Google Fonts のグリフアウトラインデータから A–Z の 26 文字を分類するモデルを学習します。

## モデル構成

フォントのアウトライン（描画コマンド列）を入力とし、ModernBERT ベースの Transformer で分類を行います。

| 項目 | 値 |
| --- | --- |
| エンコーダ | ModernBERT (HuggingFace Transformers) |
| 入力表現 | 描画コマンド種別の one-hot + 座標を線形射影 |
| プーリング | CLS トークン |
| クラス数 | 26 (A–Z) |
| スケジューラ | Warmup + Cosine Annealing |

## プロジェクト構成

```
src/classification/
├── train.py            # 学習エントリーポイント
├── module.py           # FontClassifier / CommandEmbedding
├── lit_module.py       # LightningModule (学習・評価ロジック)
├── lit_data_module.py  # LightningDataModule (データ読込・分割)
└── optim.py            # WarmupCosineAnnealingLR
```

## セットアップ

[uv](https://docs.astral.sh/uv/) が必要です。

```bash
uv sync
```

### Dev Container

VS Code の Dev Containers 拡張機能を使えば、環境構築を自動化できます。
GPU がある場合は自動的に利用されます。

## 学習の実行

```bash
uv run python -m classification.train
```

RTX 3060 Ti 程度の GPU が必要です。
学習時間は 1 分半程度です。

初回実行時に Google Fonts データセットのダウンロードが行われるため、時間がかかる場合があります。

学習が完了すると `lightning_logs/` 以下に次のファイルが出力されます。

- `test_confusion_matrix.pdf` — 混同行列の可視化
- `test_classification_report.txt` — クラスごとの Precision / Recall / F1
- `checkpoints/` — モデルのチェックポイント

TensorBoard でログを確認できます。

```bash
uv run tensorboard --logdir lightning_logs
```

## 実験結果

1 エポックの学習で、テストデータに対して全体の accuracy **90%** を達成しました。
各文字の F1 スコアは 0.82–0.96 の範囲にあり、マクロ平均の Precision / Recall / F1 はいずれも 0.90 です。

詳細は [`results/`](results/) を参照してください。
