import torch

# Load predictions from the three checkpoints
p5  = torch.load("preds_epoch5_fliptta.pt")
p8  = torch.load("preds_epoch8_fliptta.pt")
p10 = torch.load("preds_epoch10_fliptta.pt")

assert len(p5) == len(p8) == len(p10)

ensemble = []
for d5, d8, d10 in zip(p5, p8, p10):
    # Make sure they line up
    nid = d5["narration_id"]
    assert nid == d8["narration_id"] == d10["narration_id"]

    v5,  n5  = d5["verb_output"],  d5["noun_output"]
    v8,  n8  = d8["verb_output"],  d8["noun_output"]
    v10, n10 = d10["verb_output"], d10["noun_output"]

    # Simple equal-weight average (you can tweak weights if you want)
    verb_ens = (v5 + v8 + v10) / 3.0
    noun_ens = (n5 + n8 + n10) / 3.0

    ensemble.append({
        "narration_id": nid,
        "verb_output": verb_ens,
        "noun_output": noun_ens,
    })

torch.save(ensemble, "submission_ens_5_8_10.pt")
print("Saved ensemble to submission_ens_5_8_10.pt")
