"""Script to apply INT8 quantization to the ASVD-compressed CNN."""

import json
import logging
from pathlib import Path

import torch
import torch.ao.quantization as quant
from dotenv import load_dotenv
from torch.ao.quantization.qconfig import get_default_qconfig

from src.ETL_data.loader import get_dataloaders
from src.model.loss import FocalLoss
from src.training.optimizer_benchmark import compute_balanced_alpha, labels_from_loader
from src.training.utils import evaluate_model

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_quantized() -> None:
    """Evaluates the statically quantized INT8 model.
    
    Loads the pre-trained ASVD-compressed model (saved by train_asvd.py),
    applies Post-Training Static Quantization (PTQ) to convert FP32 weights to INT8,
    calibrates the model with training data, evaluates its performance, and
    exports it to ONNX format.
    """
    logger.info("Starting PTQ quantization and evaluation...")
    
    device = torch.device("cpu")
    
    asvd_model_path = Path("evaluation/asvd_cnn/model.pt")
    if not asvd_model_path.exists():
        logger.error(f"Pre-trained ASVD model not found at {asvd_model_path}. Run train_asvd.py first.")
        return
    
    model_fp32 = torch.load(asvd_model_path, map_location=device, weights_only=False)
    model_fp32.eval()
    
    backend = torch.backends.quantized.engine
    if backend not in ['qnnpack', 'fbgemm', 'x86']:
        backend = 'qnnpack'
        try:
            torch.backends.quantized.engine = backend
        except RuntimeError:
            pass
            
    model_fp32.qconfig = get_default_qconfig(backend)
    
    quant.fuse_modules(model_fp32, [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'], ['conv4', 'bn4']], inplace=True)
    
    model_prepared = quant.prepare(model_fp32)
    
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32)
    
    logger.info("Calibrating model...")
    calib_batches = 10
    with torch.no_grad():
        for i, (X, _) in enumerate(train_dl):
            if i >= calib_batches:
                break
            model_prepared(X)
            
    logger.info("Converting to INT8...")
    model_int8 = quant.convert(model_prepared)
    
    labels = labels_from_loader(train_dl)
    alpha = compute_balanced_alpha(labels)
    
    gamma = 2.0
    bayesian_metrics_path = Path("evaluation/bayesian_cnn/metrics.json")
    if bayesian_metrics_path.exists():
        with open(bayesian_metrics_path) as f:
            b_metrics = json.load(f)
            gamma = b_metrics.get("best_params", {}).get("gamma", 2.0)
            
    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    
    logger.info("Evaluating INT8 model...")
    test_loss, test_metrics = evaluate_model(model_int8, test_dl, loss_fn, device)
    
    metrics = {
        "test_loss": test_loss,
        **test_metrics,
        "compression": "PTQ INT8"
    }
    
    out_dir = Path("evaluation/quantized_cnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    scripted_model = torch.jit.script(model_int8)
    scripted_model.save(out_dir / "model.pt")
    logger.info(f"Saved INT8 TorchScript model to {out_dir / 'model.pt'}")

if __name__ == "__main__":
    evaluate_quantized()
