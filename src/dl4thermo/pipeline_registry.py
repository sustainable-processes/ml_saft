"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from dl4thermo.pipelines import cosmo_rs as crs
from dl4thermo.pipelines import data_processing as dp
from dl4thermo.pipelines import pc_saft_fitting as pcsaft
from dl4thermo.pipelines import result_analysis as results
from dl4thermo.pipelines import train_models as train


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    # Data Processing
    thermoml_processing_pipeline = dp.create_thermoml_pipeline()
    dortmund_processing_pipeline = dp.create_ddb_resolve_pipeline()
    dortmund_model_preparation_pipeline = (
        dp.create_dortmund_model_preparation_pipeline()
    )
    cosmo_rs = crs.create_cosmo_pipeline()
    wiki_scrape_pipeline = dp.create_wiki_scrape_pipeline()
    crc_pipeline = dp.create_crc_pipeline()
    generate_pcsaft_emulator_data = train.create_pc_saft_data_generation_pipeline()

    # PC-SAFT regression
    sensitivity_analysis_pipeline = pcsaft.create_sensitivity_analysis_pipeline()
    pcsaft_regression_pipeline = pcsaft.create_pure_component_regression_pipeline()
    pcsaft_cosmo_regression_pipeline = (
        pcsaft.create_pure_component_cosmo_regression_pipeline()
    )

    # Splitting
    prepare_split = train.create_sepp_splits_pipeline()
    prepare_regressed_split = train.create_regressed_splits_pipeline()
    prepare_sigma_moments_split = train.create_sigma_moments_splits_pipeline()
    prepare_e2e_split = train.create_e2e_splits()

    # Training
    train_pc_saft_emulator_pvap_density = (
        train.create_pc_saft_emulator_pvap_density_pipeline()
    )
    train_pc_saft_emulator_critical_model = (
        train.create_pc_saft_emulator_critical_pipeline()
    )
    train_sklearn_sepp_pcp_saft_model = (
        train.create_sklearn_sepp_pcp_saft_model_pipeline()
    )
    train_sklearn_regressed_pcp_saft_model = (
        train.create_sklearn_regressed_pcp_saft_model_pipeline()
    )
    train_sklearn_cv_regressed_pcp_saft_model = (
        train.create_sklearn_cv_regressed_pcp_saft_model_pipeline()
    )
    train_lolo_regressed_pcp_saft_model = (
        train.create_lolo_regressed_pcp_saft_model_pipeline()
    )
    train_ffn_sepp_pcp_saft_model = train.create_ffn_sepp_pcp_saft_model_pipeline()
    train_ffn_regressed_pcp_saft_model = (
        train.create_ffn_regressed_pcp_saft_model_pipeline()
    )
    train_ffn_cv_regressed_pcp_saft_model = (
        train.create_ffn_cv_regressed_pcp_saft_model_pipeline()
    )
    train_chemprop_regressed_pcp_saft_model = (
        train.create_chemprop_regressed_pcp_saft_model_pipeline()
    )
    train_chemprop_cv_regressed_pcp_saft_model = (
        train.create_chemprop_cv_regressed_pcp_saft_model_pipeline()
    )
    train_chemprop_sepp_pcp_saft_model = (
        train.create_chemprop_sepp_pcp_saft_model_pipeline()
    )
    train_pyg_sepp_pcp_saft_model = train.create_pyg_sepp_pcp_saft_model_pipeline()
    train_pyg_regressed_pcp_saft_model = (
        train.create_pyg_regressed_pcp_saft_model_pipeline()
    )
    train_pyg_cv_regressed_pcp_saft_model = (
        train.create_pyg_cv_regressed_pcp_saft_model_pipeline()
    )
    train_pyg_pretrain_cosmo_model_pipeline = (
        train.create_pyg_pretrain_cosmo_model_pipeline()
    )
    training_spk_sepp_pcp_saft_model_pipeline = (
        train.create_spk_sepp_pcp_saft_model_pipeline()
    )
    training_spk_rdkit_sepp_pcp_saft_model_pipeline = (
        train.create_spk_rdkit_sepp_pcp_saft_model_pipeline()
    )
    # train_spk_regressed_pcp_saft_model = (
    #     train.create_spk_regressed_pcp_saft_model_pipeline()
    # )
    train_spk_rdkit_regressed_pcp_saft_model = (
        train.create_spk_rdkit_regressed_pcp_saft_model_pipeline()
    )
    training_sigma_moments_spk_pipeline = train.create_spk_sigma_moments_pipeline()
    train_qm9_spk_pipeline = train.create_spk_qm9_pretrained_mu_model_pipeline()
    train_spk_sepp_mu_model_pipeline = train.create_spk_sepp_mu_model_pipeline()
    train_spk_qm9_mixed_mu_model_pipeline = (
        train.create_spk_qm9_mixed_mu_model_pipeline()
    )

    # Results and analysis
    pcsaft_fitting_results_table_pipeline = (
        results.create_pcsaft_regression_results_table_pipeline()
    )
    pcsaft_cosmo_fitting_results_table_pipeline = (
        results.create_cosmo_pcsaft_regression_results_table_pipeline()
    )
    results_table_pipeline = results.create_results_table_pipeline()
    cosmo_pretrain_results_table_pipeline = (
        results.create_cosmo_pretrain_results_table_pipeline()
    )

    return {
        "__default__": dortmund_processing_pipeline
        + dortmund_model_preparation_pipeline
        + pcsaft_regression_pipeline
        # + prepare_split
        + pcsaft_fitting_results_table_pipeline
        + prepare_regressed_split
        + pcsaft_cosmo_fitting_results_table_pipeline
        # + train_sklearn_sepp_pcp_saft_model
        + train_sklearn_regressed_pcp_saft_model
        # + train_ffn_sepp_pcp_saft_model
        + train_ffn_regressed_pcp_saft_model
        # + train_chemprop_sepp_pcp_saft_model
        + train_chemprop_regressed_pcp_saft_model
        # + train_pyg_sepp_pcp_saft_model
        + train_pyg_regressed_pcp_saft_model
        # + training_spk_sepp_pcp_saft_model_pipeline
        # + training_spk_rdkit_sepp_pcp_saft_model_pipeline
        # + train_spk_sepp_mu_model_pipeline
        + train_spk_rdkit_regressed_pcp_saft_model + results_table_pipeline,
        "wiki_scrape": wiki_scrape_pipeline,
        "thermoml": thermoml_processing_pipeline,
        "ddb": dortmund_processing_pipeline,
        "ddb_model_prep": dortmund_model_preparation_pipeline,
        "crc": crc_pipeline,
        "dp": dortmund_processing_pipeline,
        "cosmo": cosmo_rs,
        "pcsaft_regression": pcsaft_regression_pipeline,
        "pcsaft_cosmo_regression": pcsaft_cosmo_regression_pipeline,
        "generate_pcsaft_emulator_data": generate_pcsaft_emulator_data,
        "train_pc_saft_emulator_pvap_density": train_pc_saft_emulator_pvap_density,
        "train_pc_saft_emulator_critical_model": train_pc_saft_emulator_critical_model,
        "prepare_sepp_split": prepare_split,
        "prepare_regressed_split": prepare_regressed_split,
        "prepare_sigma_moments_split": prepare_sigma_moments_split,
        "prepare_e2e_split": prepare_e2e_split,
        "train_sklearn_sepp_model": train_sklearn_sepp_pcp_saft_model,
        "train_sklearn_regressed_model": train_sklearn_regressed_pcp_saft_model,
        "train_sklearn_cv_regressed_model": train_sklearn_cv_regressed_pcp_saft_model,
        "train_lolo_regressed_model": train_lolo_regressed_pcp_saft_model,
        "train_ffn_sepp_model": train_ffn_sepp_pcp_saft_model,
        "train_ffn_regressed_model": train_ffn_regressed_pcp_saft_model,
        "train_ffn_cv_regressed_model": train_ffn_cv_regressed_pcp_saft_model,
        "train_chemprop_sepp_model": train_chemprop_sepp_pcp_saft_model,
        "train_chemprop_cv_regressed_model": train_chemprop_cv_regressed_pcp_saft_model,
        "train_chemprop_regressed_model": train_chemprop_regressed_pcp_saft_model,
        "train_pyg_sepp_model": train_pyg_sepp_pcp_saft_model,
        "train_pyg_regressed_model": train_pyg_regressed_pcp_saft_model,
        "train_pyg_cv_regressed_model": train_pyg_cv_regressed_pcp_saft_model,
        "train_pyg_pretrain_cosmo_model": train_pyg_pretrain_cosmo_model_pipeline,
        "all_regressed": train_sklearn_regressed_pcp_saft_model
        + train_ffn_regressed_pcp_saft_model
        + train_chemprop_regressed_pcp_saft_model
        + train_pyg_regressed_pcp_saft_model,
        "all_cv": train_sklearn_cv_regressed_pcp_saft_model
        + train_ffn_cv_regressed_pcp_saft_model
        + train_chemprop_cv_regressed_pcp_saft_model
        + train_pyg_cv_regressed_pcp_saft_model,
        "train_spk_model": training_spk_sepp_pcp_saft_model_pipeline,
        "train_spk_rdkit_model": training_spk_rdkit_sepp_pcp_saft_model_pipeline,
        "train_spk_rdkit_regressed_model": train_spk_rdkit_regressed_pcp_saft_model,
        "train_spk_mu_model": train_spk_sepp_mu_model_pipeline,
        "train_spk_sigma_moments": training_sigma_moments_spk_pipeline,
        "train_spk_qm9": train_qm9_spk_pipeline,
        "train_spk_qm9_mixed_model": train_spk_qm9_mixed_mu_model_pipeline,
        "sensitivity_analysis": sensitivity_analysis_pipeline,
        "pcp_saft_fitting_results_table": pcsaft_fitting_results_table_pipeline,
        "pcp_saft_cosmo_fitting_results_table": pcsaft_cosmo_fitting_results_table_pipeline,
        "results_table": results_table_pipeline,
        "cosmo_pretrain_results_table": cosmo_pretrain_results_table_pipeline,
    }
