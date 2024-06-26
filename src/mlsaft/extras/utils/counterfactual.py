import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs


def calculate_ecfp_from_smiles(smiles):
    return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))  # type: ignore


def calculate_tanimoto_similarity(smiles_1, smiles_2):
    ecfp_1 = calculate_ecfp_from_smiles(smiles_1)
    ecfp_2 = calculate_ecfp_from_smiles(smiles_2)
    return DataStructs.TanimotoSimilarity(ecfp_1, ecfp_2)  # type: ignore


def get_counterfactual_from_list(
    smiles,
    value,
    compare_smiles,
    compare_values,
    similarity_metric="Tanimoto",
    min_similarity=None,
    max_value_diff=None,
):
    compared_smiles = []
    compared_values = []
    similarities = []
    value_diffs = []
    for cs_idx, cs in enumerate(compare_smiles):
        # do not include pairs of same struture
        if smiles == cs:
            continue
        if similarity_metric == "Tanimoto":
            sim = calculate_tanimoto_similarity(smiles, cs)
        else:
            raise NotImplementedError(
                f"Similarity metric {similarity_metric} not implemented."
            )
        # directly ignore if similarity threshold is not met
        if min_similarity is not None:
            if sim < min_similarity:
                continue
        value_diff = value - compare_values[cs_idx]
        # directly ignore if value threshold is not met
        if max_value_diff is not None:
            if value_diff > max_value_diff:
                continue
        compared_smiles.append(cs)
        compared_values.append(compare_values[cs_idx])
        similarities.append(sim)
        value_diffs.append(value_diff)

    s_counterfactuals = {
        "smiles": smiles,
        "value": value,
        "compared_smiles": np.array(compared_smiles),
        "compared_value": np.array(compare_values),
        "similarity": np.array(similarities),
        "value_difference": np.array(value_diffs),
    }
    return s_counterfactuals


def counterfactual_analysis(
    smiles,
    predictions,
    compare_smiles,
    compare_values,
    type_simMol_diffValue_lb_value_diff=None,
    type_diffMol_simValue_ub_value_diff=None,
    type_max_simMol_diffValue_lambda=None,
    type_min_simMol_diffValue_lambda=None,
):
    counterfactual_dict = {}
    print(
        f"Starting counterfactual analysis of {len(smiles)} molecules. This might take a while."
    )
    for s_idx, s in enumerate(smiles):
        if s_idx + 1 % 100 == 0:
            print(f"Mol #{s_idx}")
        counterfactual_dict[s] = get_counterfactual_from_list(
            s, predictions[s_idx], compare_smiles, compare_values
        )

    # TYPE_simMol_diffValue: get all counterfactual sorted according to descending similarity and lower bound on value diff, i.e., molecules with high similaritiy and a minimal amount of value difference (even though they have high similarity the value is highly different)
    if type_simMol_diffValue_lb_value_diff is not None:
        all_counterfactual_dict = {}
        all_smiles = []
        all_value_diffs = []
        all_similarities = []
        all_values = []
        # get counterfactual per mol
        for s, s_counterf_dict in counterfactual_dict.items():
            # get only values with diff > lb
            abs_comp_value_diffs = np.abs(
                s_counterf_dict["value_difference"]
            )  # TODO: ABS
            crit_indices = np.where(
                abs_comp_value_diffs > type_simMol_diffValue_lb_value_diff
            )
            # get sording indices according to similarity (descending)
            comp_sim = s_counterf_dict["similarity"][crit_indices]
            sort_indices = np.argsort(-1 * comp_sim)
            # extract and sort all
            extracted_value_differences = s_counterf_dict["value_difference"][
                crit_indices
            ][sort_indices]
            extracted_similarities = comp_sim[sort_indices]
            extracted_comp_smiles = s_counterf_dict["compared_smiles"][crit_indices][
                sort_indices
            ]
            extracted_comp_value = s_counterf_dict["compared_value"][crit_indices][
                sort_indices
            ]
            all_counterfactual_dict[s] = {
                "smiles": s_counterf_dict["smiles"],
                "value": s_counterf_dict["value"],
                "compared_smiles": extracted_comp_smiles,
                "compared_value": extracted_comp_value,
                "similarity": extracted_similarities,
                "value_difference": extracted_value_differences,
            }
            all_smiles += [
                f"{s_counterf_dict['smiles']}_{cs}" for cs in extracted_comp_smiles
            ]
            all_values += [
                f"{s_counterf_dict['value']}_{cv}" for cv in extracted_comp_value
            ]
            all_value_diffs += list(extracted_value_differences)
            all_similarities += list(extracted_similarities)

        # get overall list of countercatual pairs
        sort_indices = np.argsort(-1 * np.array(all_similarities))
        all_smiles = np.array(all_smiles)[sort_indices]
        all_values = np.array(all_values)[sort_indices]
        all_value_diffs = np.array(all_value_diffs)[sort_indices]
        all_similarities = np.array(all_similarities)[sort_indices]
        df_all_TYPE_simMol_diffValue = pd.DataFrame(
            {
                "SMILES-Pair": all_smiles,
                "Value-Pair": all_values,
                "Value_diff": all_value_diffs,
                "Similarity": all_similarities,
            }
        )
    else:
        df_all_TYPE_simMol_diffValue = None

    # TYPE_diffMol_simValue: get all counterfactuals sorted according to ascending similarity and upper bound on value diff, i.e., molecules with low similaritiy and a maximal amount of value difference (even though they have low similarity the value is very similar)
    if type_diffMol_simValue_ub_value_diff is not None:
        all_counterfactual_dict = {}
        all_smiles = []
        all_value_diffs = []
        all_similarities = []
        all_values = []
        for s, s_counterf_dict in counterfactual_dict.items():
            # get only values with diff < ub
            abs_comp_value_diffs = np.abs(
                s_counterf_dict["value_difference"]
            )  # TODO: ABS
            crit_indices = np.where(
                abs_comp_value_diffs < type_diffMol_simValue_ub_value_diff
            )
            # get sording indices according to similarity (ascending)
            comp_sim = s_counterf_dict["similarity"][crit_indices]
            sort_indices = np.argsort(comp_sim)
            # extract and sort all
            extracted_value_differences = s_counterf_dict["value_difference"][
                crit_indices
            ][sort_indices]
            extracted_similarities = comp_sim[sort_indices]
            extracted_comp_smiles = s_counterf_dict["compared_smiles"][crit_indices][
                sort_indices
            ]
            extracted_comp_value = s_counterf_dict["compared_value"][crit_indices][
                sort_indices
            ]
            all_counterfactual_dict[s] = {
                "smiles": s_counterf_dict["smiles"],
                "value": s_counterf_dict["value"],
                "compared_smiles": extracted_comp_smiles,
                "compared_value": extracted_comp_value,
                "similarity": extracted_similarities,
                "value_difference": extracted_value_differences,
            }
            all_smiles += [
                f"{s_counterf_dict['smiles']}_{cs}" for cs in extracted_comp_smiles
            ]
            all_values += [
                f"{s_counterf_dict['value']}_{cv}" for cv in extracted_comp_value
            ]
            all_value_diffs += list(extracted_value_differences)
            all_similarities += list(extracted_similarities)

        # get overall list of countercatual pairs
        sort_indices = np.argsort(all_similarities)
        all_smiles = np.array(all_smiles)[sort_indices]
        all_values = np.array(all_values)[sort_indices]
        all_value_diffs = np.array(all_value_diffs)[sort_indices]
        all_similarities = np.array(all_similarities)[sort_indices]
        df_all_TYPE_diffMol_simValue = pd.DataFrame(
            {
                "SMILES-Pair": all_smiles,
                "Value-Pair": all_values,
                "Value_diff": all_value_diffs,
                "Similarity": all_similarities,
            }
        )
    else:
        df_all_TYPE_diffMol_simValue = None

    # TYPE_max_simMol_diffValue: get #<num_counterfactuals> wrt. max similarity + lambda * value diff
    # cf. Eq. 7 in Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R., & Zavala, V. (2022). Capturing Molecular Interactions in Graph Neural Networks: A Case Study in Multi-Component Phase Equilibrium.
    if type_max_simMol_diffValue_lambda is not None:
        all_counterfactual_dict = {}
        all_smiles = []
        all_value_diffs = []
        all_similarities = []
        all_values = []
        all_crit_values = []
        # get counterfactual per mol
        for s, s_counterf_dict in counterfactual_dict.items():
            # get only values with diff > lb
            abs_comp_value_diffs = np.abs(
                s_counterf_dict["value_difference"]
            )  # TODO: ABS
            comp_sim = s_counterf_dict["similarity"]
            sort_crit_values = (
                comp_sim + type_max_simMol_diffValue_lambda * abs_comp_value_diffs
            )
            # get sording indices according to similarity (descending)
            sort_indices = np.argsort(-1 * sort_crit_values)
            # extract and sort all
            sort_crit_values = sort_crit_values[sort_indices]
            extracted_value_differences = s_counterf_dict["value_difference"][
                sort_indices
            ]
            extracted_similarities = comp_sim[sort_indices]
            extracted_comp_smiles = s_counterf_dict["compared_smiles"][sort_indices]
            extracted_comp_value = s_counterf_dict["compared_value"][sort_indices]
            all_counterfactual_dict[s] = {
                "smiles": s_counterf_dict["smiles"],
                "value": s_counterf_dict["value"],
                "compared_smiles": extracted_comp_smiles,
                "compared_value": extracted_comp_value,
                "similarity": extracted_similarities,
                "value_difference": extracted_value_differences,
                "crit_values": sort_crit_values,
            }
            all_smiles += [
                f"{s_counterf_dict['smiles']}_{cs}" for cs in extracted_comp_smiles
            ]
            all_values += [
                f"{s_counterf_dict['value']}_{cv}" for cv in extracted_comp_value
            ]
            all_value_diffs += list(extracted_value_differences)
            all_similarities += list(extracted_similarities)
            all_crit_values += list(sort_crit_values)

        # get overall list of countercatual pairs
        sort_indices = np.argsort(-1 * np.array(all_crit_values))
        all_smiles = np.array(all_smiles)[sort_indices]
        all_values = np.array(all_values)[sort_indices]
        all_value_diffs = np.array(all_value_diffs)[sort_indices]
        all_similarities = np.array(all_similarities)[sort_indices]
        df_all_TYPE_max_simMol_diffValue = pd.DataFrame(
            {
                "SMILES-Pair": all_smiles,
                "Value-Pair": all_values,
                "Value_diff": all_value_diffs,
                "Similarity": all_similarities,
            }
        )
    else:
        df_all_TYPE_max_simMol_diffValue = None

    # TYPE_min_simMol_diffValue: get #<num_counterfactuals> wrt: min similarity + lambda * value diff
    # cf. Eq. 8 in Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R., & Zavala, V. (2022). Capturing Molecular Interactions in Graph Neural Networks: A Case Study in Multi-Component Phase Equilibrium.
    if type_min_simMol_diffValue_lambda is not None:
        all_counterfactual_dict = {}
        all_smiles = []
        all_value_diffs = []
        all_similarities = []
        all_values = []
        all_crit_values = []
        sort_crit_values = ()
        # get counterfactual per mol
        for s, s_counterf_dict in counterfactual_dict.items():
            # get only values with diff > lb
            abs_comp_value_diffs = np.abs(
                s_counterf_dict["value_difference"]
            )  # TODO: ABS
            comp_sim = s_counterf_dict["similarity"]
            sort_crit_values = (
                comp_sim + type_min_simMol_diffValue_lambda * abs_comp_value_diffs
            )
            # get sording indices according to similarity (descending)
            sort_indices = np.argsort(-1 * sort_crit_values)
            # extract and sort all
            sort_crit_values = sort_crit_values[sort_indices]
            extracted_value_differences = s_counterf_dict["value_difference"][
                sort_indices
            ]
            extracted_similarities = comp_sim[sort_indices]
            extracted_comp_smiles = s_counterf_dict["compared_smiles"][sort_indices]
            extracted_comp_value = s_counterf_dict["compared_value"][sort_indices]
            all_counterfactual_dict[s] = {
                "smiles": s_counterf_dict["smiles"],
                "value": s_counterf_dict["value"],
                "compared_smiles": extracted_comp_smiles,
                "compared_value": extracted_comp_value,
                "similarity": extracted_similarities,
                "value_difference": extracted_value_differences,
                "crit_values": sort_crit_values,
            }
            all_smiles += [
                f"{s_counterf_dict['smiles']}_{cs}" for cs in extracted_comp_smiles
            ]
            all_values += [
                f"{s_counterf_dict['value']}_{cv}" for cv in extracted_comp_value
            ]
            all_value_diffs += list(extracted_value_differences)
            all_similarities += list(extracted_similarities)
        all_crit_values += list(sort_crit_values)

        # get overall list of countercatual pairs
        sort_indices = np.argsort(-1 * np.array(all_crit_values))
        all_smiles = np.array(all_smiles)[sort_indices]
        all_values = np.array(all_values)[sort_indices]
        all_value_diffs = np.array(all_value_diffs)[sort_indices]
        all_similarities = np.array(all_similarities)[sort_indices]
        df_all_TYPE_min_simMol_diffValue = pd.DataFrame(
            {
                "SMILES-Pair": all_smiles,
                "Value-Pair": all_values,
                "Value_diff": all_value_diffs,
                "Similarity": all_similarities,
            }
        )
    else:
        df_all_TYPE_min_simMol_diffValue = None

    return (
        df_all_TYPE_simMol_diffValue,
        df_all_TYPE_diffMol_simValue,
        df_all_TYPE_max_simMol_diffValue,
        df_all_TYPE_min_simMol_diffValue,
    )
