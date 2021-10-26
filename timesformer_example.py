import torch
from TimeSformer.timesformer.models.vit import TimeSformer

model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time')

dummy_video = torch.randn(2, 8, 3, 224, 224) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 400)
