# from pura.resolvers import resolve_identifiers
# from pura.compound import CompoundIdentifierType
# from rdkit import Chem
# from rdkit.Chem.Draw import MolsToGridImage
# from chemprop.train.make_predictions import make_predictions
# from chemprop.args import PredictArgs
# import wandb
# import pandas as pd
# from pathlib import Path
# import ast

# molecules = [
#     # Pure
#     "CO2",
#     "Hexane",
#     "1-Hexene",
#     "1-Hexyne",
#     "cyclohexane",
#     "benzene",
#     "eicosane",
#     "2,3,3-trimethylpentane",
#     "2,2,4-trimethylpentane",
#     # Diploar
#     "hexanal",
#     "dimethylsulfoxide",
#     "methylbutanoate",
#     # Associative
#     "Methanol",
#     "Hexanol",
#     "ethylamine",
#     "acetic acid",
# ]

# # Get smiles
# results = resolve_identifiers(molecules, CompoundIdentifierType.SMILES)
# smiles_list = [
#     (input_compound.identifiers[0].value, output_identifiers[0].value)
#     for input_compound, output_identifiers in results
# ]

# # RDkit molecules
# img = MolsToGridImage(
#     [Chem.MolFromSmiles(smiles) for name, smiles in smiles_list],
#     legends=[name for name, smiles in smiles_list],
#     molsPerRow=4,
# )
# img.save("data/08_reporting/example_compounds.png")

# # Load model
# api = wandb.Api()
# artifact = api.artifact("ceb-sre/dl4thermo/chemprop_sepp:latest")
# root = Path("data/07_model_output/")
# artifact.download(root=root)

# # Make predictions
# args = PredictArgs()
# args.preds_path = "data/08_reporting/example_compounds_preds.csv"
# args.test_path = "data/08_reporting/example_compounds_preds.csv"
# args.checkpoint_path = root / "fold_0/model_0/model.pt"
# args.num_workers = 0
# args.batch_size = 1
# args.process_args()
# preds = make_predictions(args=args, smiles=[[smiles] for _, smiles in smiles_list])


# # Add names
# df = pd.read_csv(args.preds_path, sep=",")
# smiles_dict = {smiles: name for name, smiles in smiles_list}
# df["smiles"] = df["smiles"].apply(lambda x: ast.literal_eval(x))
# df["name"] = [smiles_dict[smiles_l[0]] for smiles_l in df["smiles"].tolist()]
# df.to_csv(args.preds_path, index=False)
