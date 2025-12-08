import torch
preds = torch.load("submission.pt")
print(len(preds))
print(preds[0].keys())
print(preds[0]["verb_output"].shape, preds[0]["noun_output"].shape)
