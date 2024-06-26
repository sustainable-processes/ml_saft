"""
This is a boilerplate pipeline 'train_models'
generated using Kedro 0.18.0
"""

from .models.chemprop import (
    ChempropDataModule,
    ChempropLightningModule,
    ChempropTrainArgs,
)
from .models.pyg import PyGDataModule, PyGLightningModule, PyGTrainArgs
from .models.spk import SPKDataModule, SPKTrainArgs
from .pipeline import (  # create_spk_regressed_pcp_saft_model_pipeline,
    create_base_pyg_cv_train_pipeline,
    create_chemprop_cv_regressed_pcp_saft_model_pipeline,
    create_chemprop_regressed_pcp_saft_model_pipeline,
    create_chemprop_sepp_pcp_saft_model_pipeline,
    create_e2e_splits,
    create_ffn_cv_regressed_pcp_saft_model_pipeline,
    create_ffn_regressed_pcp_saft_model_pipeline,
    create_ffn_sepp_pcp_saft_model_pipeline,
    create_lolo_regressed_pcp_saft_model_pipeline,
    create_pc_saft_data_generation_pipeline,
    create_pc_saft_emulator_critical_pipeline,
    create_pc_saft_emulator_pvap_density_pipeline,
    create_pyg_cv_regressed_pcp_saft_model_pipeline,
    create_pyg_pretrain_cosmo_model_pipeline,
    create_pyg_regressed_pcp_saft_model_pipeline,
    create_pyg_sepp_pcp_saft_model_pipeline,
    create_regressed_splits_pipeline,
    create_sepp_splits_pipeline,
    create_sigma_moments_splits_pipeline,
    create_sklearn_cv_regressed_pcp_saft_model_pipeline,
    create_sklearn_regressed_pcp_saft_model_pipeline,
    create_sklearn_sepp_pcp_saft_model_pipeline,
    create_spk_qm9_mixed_mu_model_pipeline,
    create_spk_qm9_pretrained_mu_model_pipeline,
    create_spk_rdkit_regressed_pcp_saft_model_pipeline,
    create_spk_rdkit_sepp_pcp_saft_model_pipeline,
    create_spk_sepp_mu_model_pipeline,
    create_spk_sepp_pcp_saft_model_pipeline,
    create_spk_sigma_moments_pipeline,
)

__version__ = "0.1"
