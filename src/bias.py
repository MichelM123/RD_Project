import torch

preds = torch.load("submission_conf_ens.pt")

alpha = 0.3   

for d in preds:
    v = d["verb_output"]
    v[0] = v[0] - alpha   
    d["verb_output"] = v

torch.save(preds, "submission.pt")
print("saved submission.pt")
