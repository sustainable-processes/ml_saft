import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import wandb
from wandb.apis.public import Run


def download_runs_wandb(
    api: wandb.Api,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> List[Run]:
    """Download runs from wandb

    Parameters
    ----------
    api : wandb.Api
        The wandb API object.
    wandb_entity : str, optional
        The wandb entity to search, by default "ceb-sre"
    wandb_project : str, optional
        The wandb project to search, by default "multitask"
    include_tags : Optional[List[str]], optional
        A list of tags that the run must have, by default None
    filter_tags : Optional[List[str]], optional
        A list of tags that the run must not have, by default None
    extra_filters : Optional[Dict[str, Any]], optional
        A dictionary of extra filters to apply to the wandb search, by default None
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading runs from wandb")

    # Filters
    filters = {}
    tag_query = []
    if include_tags is not None and len(include_tags) > 0:
        for include_tag in include_tags:
            tag_query.append({"tags": {"$in": [include_tag]}})
        # filters["tags"] = {"$infilt": include_tags}
    if filter_tags is not None and len(filter_tags) > 0:
        tag_query += [{"tags": {"$nin": filter_tags}}]
    if len(tag_query) > 0:
        filters["$and"] = tag_query
    if only_finished_runs:
        filters["state"] = "finished"
    if extra_filters is not None:
        filters.update(extra_filters)

    # Get runs
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        filters=filters,
    )
    return runs


def download_best_model(run) -> Tuple[Path, str]:
    """Download best model from wandb

    Arguments
    ----------


    Returns
    -------
    Path to best model checkpoint

    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading best model from wandb")
    # Get best checkpoint
    artifacts = run.logged_artifacts()  # type: ignore
    ckpt_path = None
    artifact_name = None
    for artifact in artifacts:
        if artifact.type == "model" and "best_k" in artifact.aliases:
            ckpt_path = artifact.download()
            artifact_name = artifact.name
    if ckpt_path is None or artifact_name is None:
        raise ValueError("No best checkpoint found")
    ckpt_path = Path(ckpt_path) / "model.ckpt"

    return ckpt_path, artifact_name
