import logging
from pathlib import Path

import pandas as pd
from joblib import load

from dl4thermo.extras.utils.molecular_fingerprints import compute_morgan_fingerprints
from wandb import Api


def main(smiles_list_path: str, save_path: str = "predictions.csv"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Download wandb artifact
    logger.info("Downloading model from wandb")
    api = Api()
    run = api.run("ceb-sre/dl4thermo/2eftwbx2")
    artifacts = run.logged_artifacts()
    model_path = None
    for artifact in artifacts:
        if artifact.type == "model":
            model_path = artifact.download()
    if model_path is None:
        raise ValueError("Model not found")
    model_path = Path(model_path) / "model.pkl"

    # Load model
    logger.info("Loading model")
    with open(model_path, "rb") as f:
        model = load(f)

    # Load smiles
    logger.info("Loading smiles")
    with open(smiles_list_path, "r") as f:
        smiles_list = f.readlines()

    # Create fingerprints
    logger.info("Computing fingerprints")
    fps = compute_morgan_fingerprints(smiles_list)

    # Make prediction
    target_columns = ["m", "sigma", "epsilon_k", "epsilonAB", "KAB"]
    logger.info("Making predictions")
    preds = model.predict(fps)
    df = pd.DataFrame(preds, columns=target_columns)

    # Save predictions to csv
    logger.info("Saving predictions")
    df["smiles"] = smiles_list
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # Use argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_list_path", type=str)
    parser.add_argument("--save_path", type=str, default="predictions.csv")
    args = parser.parse_args()
    main(args.smiles_list_path, args.save_path)
