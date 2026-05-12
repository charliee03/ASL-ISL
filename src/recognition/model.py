import torch.nn as nn


class SignRecognitionTransformer(nn.Module):
    def __init__(self, num_keypoints=21, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, vocab_size=2000):
        super().__init__()
        self.input_proj = nn.Linear(num_keypoints * 3, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt=None):
        x = self.input_proj(x)
        if tgt is not None:
            out = self.transformer(x, tgt)
        else:
            out = self.transformer(x)
        return self.output_proj(out)
