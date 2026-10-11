"""FairFace ResNet-34, allineamento dlib e sola testa di genere (logit 7:9)."""

from pathlib import Path

from .storage import read_json, save_json, sha256, verify_images


class FairFace:
    def __init__(self, weights, detector, landmarks, device="cpu"):
        import dlib
        import torch
        from torchvision import models, transforms

        self.dlib, self.torch, self.device = dlib, torch, device
        self.detector = dlib.cnn_face_detection_model_v1(str(detector))
        self.landmarks = dlib.shape_predictor(str(landmarks))
        self.model = models.resnet34(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def classify(self, path):
        image = self.dlib.load_rgb_image(str(path))
        height, width = image.shape[:2]
        scale = 800 / max(height, width)
        image = self.dlib.resize_image(image, rows=max(1, round(height * scale)), cols=max(1, round(width * scale)))
        detections = self.detector(image, 1)
        labels, scores = [], []
        for detection in detections:
            shape = self.landmarks(image, detection.rect)
            crop = self.dlib.get_face_chip(image, shape, size=300, padding=0.25)
            tensor = self.transform(crop).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                probabilities = self.model(tensor)[0, 7:9].softmax(dim=0).cpu().tolist()
            labels.append("male" if probabilities[0] >= probabilities[1] else "female")
            scores.append(probabilities)
        return {"labels": labels, "probabilities_male_female": scores, "face_count": len(labels)}


def annotate_run(root, weights, detector, landmarks, device="cpu"):
    root = Path(root)
    state = read_json(root / "checkpoint.json")
    if state["config"]["mock"]:
        raise ValueError("FairFace non va applicato alle immagini fittizie del mock")
    verify_images(root, state)
    provenance = {"weights_sha256": sha256(weights), "detector_sha256": sha256(detector),
                  "landmarks_sha256": sha256(landmarks), "device": device,
                  "padding": 0.25, "face_chip_size": 300, "detection_max_size": 800}
    path = root / "fairface.json"
    results = read_json(path) if path.exists() else {"provenance": provenance, "images": []}
    if results["provenance"] != provenance:
        raise ValueError("La classificazione esistente usa pesi o impostazioni diverse")
    model = FairFace(weights, detector, landmarks, device)
    done = {r["path"] for r in results["images"] if not r.get("error")}
    results["images"] = [r for r in results["images"] if r["path"] in done]
    images = [image for record in state["records"] for image in record["images"]]
    for image in images:
        if image["path"] in done:
            continue
        row = {"path": image["path"], "sha256": image["sha256"]}
        try:
            row.update(model.classify(root / image["path"]))
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", labels=[])
        results["images"].append(row)
        save_json(path, results)
        print(f"FairFace: {len(results['images'])}/{len(images)}", flush=True)
