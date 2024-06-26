import logging
from abc import ABC, abstractmethod
from datetime import datetime as dt
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import ray
from tqdm import tqdm, trange


class ParallelMoleculeTask(ABC):
    @staticmethod
    @abstractmethod
    def filter_func(
        data: pd.DataFrame, smiles: str, smiles_column: str
    ) -> Tuple[Any, Dict[str, Any], bool]:
        """Filter data to get the molecule to be processed

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the molecule to be processed
        smiles : str
            SMILES string

        Returns
        -------
        Tuple[Any, Dict[str, Any], bool]
            Tuple containing the data to be processed, a dictionary of keyword arguments
            to pass to the run_func, and a boolean indicating whether the molecule should be
            skipped
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def run_func(molecule_data: Any, smiles: str) -> Any:
        """Function to run in parallel

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keyword arguments to pass to the function

        Returns
        -------
        Dict[str, Any]
            Dictionary of results
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def process_func(results: Any, **kwargs) -> Any:
        """Update results

        Parameters
        ----------
        results : Dict[str, Any]
            Dictionary of results
        kwargs : Dict[str, Any]
            Keyword arguments to pass to the function
        """
        raise NotImplementedError()


def process(task: ParallelMoleculeTask, jobs_ray: dict, batch_refs: list):
    # Wait for results
    ready_refs, _ = ray.wait(batch_refs, num_returns=len(batch_refs))  # type: ignore

    # Get results
    results_ray_list = []
    for ready_ref in ready_refs:
        # Retrieve job definition
        initialization = jobs_ray[ready_ref]

        # Get  result
        results, good = ray.get(ready_ref, timeout=4)  # type: ignore
        if not good:
            continue
        results, good = task.process_func(results, **initialization)
        if not good:
            continue
        results_ray_list.append(results)

    return results_ray_list


def parallel_runner(
    task: ParallelMoleculeTask,
    data: pd.DataFrame,
    smiles_column: str,
    batch_size: int = 100,
    filter_kwargs: Optional[Dict[str, Any]] = None,
):
    """Perform sensitivity analysis on the PC-SAFT model using ray for parallelization

    Parameters
    ----------
    filter_func : Callable
        Function to filter data. Should return a tuple of (molecule_data, extra_args, good)
        where molecule_data is to be passed into remote_func under data, extra_args is a
        dictionary of keyword args to pass to remote_func, and good is a boolean indicating
        whether the molecule should be skipped.
    run_func : Callable
        Function to run in parallel. Should return a dictionary of results. Should have the
        `ray.remote` decorator. Should take
    batch_size : int, optional
        Number of molecules to process in each batch, by default 100

    Returns
    -------
    """
    logger = logging.getLogger(__name__)
    if not ray.is_initialized():  # type: ignore
        ray.init()  # type: ignore

    # Submit jobs to cluster
    jobs_ray = {}
    smiles_list = data[smiles_column].unique()
    n_tasks = 0
    skipped_list = []
    filter_kwargs = filter_kwargs or {}
    for smiles in tqdm(smiles_list, desc="Setting up jobs"):

        # Check and filter
        molecule_data, extra_args, good = task.filter_func(
            data=data, smiles=smiles, smiles_column=smiles_column, **filter_kwargs
        )
        if not good:
            skipped_list.append(smiles)
            continue

        # Submit job
        out_ref = task.run_func.remote(
            smiles=smiles,  # type: ignore
            molecule_data=molecule_data,
            **extra_args,
        )
        job = {"smiles": smiles, **extra_args}
        jobs_ray.update({out_ref: job})
        n_tasks += 1

    # Get results in batches with checkpointing via Kedro
    n_batches = n_tasks // batch_size
    n_batches += 1 if n_tasks % batch_size != 0 else 0
    object_refs = list(jobs_ray.keys())
    batch_refs_list = [
        object_refs[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    logger.info(f"Running {n_tasks} tasks in {n_batches} batches")
    return {
        f"batch_{i}": partial(process, task, jobs_ray, batch_refs_list[i])
        for i in range(n_batches)
    }
