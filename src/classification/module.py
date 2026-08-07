import torch
from torch import Tensor, nn
from torchfont.nn import OutlineEmbedding
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertModel


class FontClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embed = OutlineEmbedding(d_model)
        modernbert_config = ModernBertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=nhead,
            intermediate_size=dim_feedforward,
            max_position_embeddings=512,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder = ModernBertModel(modernbert_config)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        ops: Tensor,
        coords: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        b = ops.size(0)
        tokens = self.embed(ops, coords)
        cls = self.cls_token.expand(b, 1, -1)
        src = torch.cat([cls, tokens], dim=1)

        cls_mask = attention_mask.new_ones(b, 1)
        attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        output = self.encoder(inputs_embeds=src, attention_mask=attention_mask)
        summary = self.norm(output.last_hidden_state[:, 0, :])
        return self.head(summary)
