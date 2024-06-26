"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import asyncio
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from ase import Atoms
from bs4 import BeautifulSoup
from pura.compound import Compound, CompoundIdentifier, CompoundIdentifierType
from pura.resolvers import CompoundResolver, resolve_identifiers
from pura.services import CAS, CIR, LocalDatabase, Opsin, PubChem, Service
from pura.services.db import load_into_database
from rdkit import Chem
from tqdm import tqdm

from dl4thermo.extras.utils.conformer import RDKitConformerRunner
from dl4thermo.extras.utils.molecular_fingerprints import _canonicalize_smiles
from dl4thermo.extras.utils.parallel import parallel_runner
from dl4thermo.extras.utils.scraping import (
    extract_table_links,
    get_smiles_dipole_moment,
)

logger = logging.getLogger(__name__)


def concat_dataframes(*args) -> pd.DataFrame:
    """Concatenate dataframes"""
    return pd.concat([arg for arg in args])


def get_thermoml_molecules(data: pd.DataFrame) -> List:
    components = data["components"].str.split("__", expand=True)
    components_list = []
    for i in range(3):
        components_list.extend(components[i].tolist())
    components_list = list(set(components_list))
    return components_list


def remove_numbers_dortmund_molecules(
    data: pd.DataFrame, molecule_columns: List[str]
) -> pd.DataFrame:
    # Remove numbers at the beginning
    for col in molecule_columns:
        data.loc[:, col] = data.loc[:, col].str.extract(r"(\d+:\s)(.+)")[1]
    return data


def get_dortmund_molecules(data: pd.DataFrame, columns: List[str]) -> List:
    molecules = pd.melt(data, value_vars=columns)["value"].unique()
    molecules = pd.Series(molecules)
    return molecules.dropna().tolist()


def combine_and_deduplicate_molecule_lists(*args) -> List[str]:
    molecules = []
    for arg in args:
        molecules.extend(arg)
    return pd.Series(molecules).unique().tolist()


def get_cas_numbers(
    ids: List[str],
    ids_to_cas: pd.DataFrame,
    id_column: str = "dortmund_id",
) -> pd.DataFrame:
    ids_to_cas = ids_to_cas.set_index(id_column)
    return ids_to_cas.reindex(ids)


async def _async_update_pura_db(data: pd.DataFrame, db_path: str):
    logger.info("Cleaning data")
    data["smiles"] = data["smiles"].apply(_canonicalize_smiles)  # type: ignore
    data = data.dropna()
    data["inchi"] = data["smiles"].apply(
        lambda smi: Chem.MolToInchi(Chem.MolFromSmiles(smi))  # type: ignore
    )
    data = data.drop_duplicates(subset=["inchi"])

    await load_into_database(
        data=data,
        db_path=db_path,
        identifier_columns=[
            ("cas_number", CompoundIdentifierType.CAS_NUMBER, False),
            ("name", CompoundIdentifierType.NAME, False),
            ("alternative_name", CompoundIdentifierType.NAME, False),
            ("smiles", CompoundIdentifierType.SMILES, True),
        ],
        inchi_column="inchi",
        update_on_conflict=False,
    )


def _update_pura_db(data: pd.DataFrame, db_path: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(_async_update_pura_db(data, db_path))


def _pura_results_to_dict(input_compound, output_identifiers):
    input_identifiers = input_compound.identifiers
    r = {}
    for identifier in input_identifiers:
        if identifier.identifier_type == CompoundIdentifierType.CUSTOM:
            r.update({"id": identifier.value})
        if (
            identifier.identifier_type == CompoundIdentifierType.NAME
            and identifier.details == "alternative_name"
        ):
            r.update({"alternative_name": identifier.value})
        elif identifier.identifier_type == CompoundIdentifierType.NAME:
            r.update({"name": identifier.value})
        if identifier.identifier_type == CompoundIdentifierType.CAS_NUMBER:
            r.update({"cas_number": identifier.value})
    smiles = None
    if isinstance(output_identifiers, list):
        if len(output_identifiers) > 0:
            if isinstance(output_identifiers[0], CompoundIdentifier):
                smiles = output_identifiers[0].value

    r.update({"smiles": smiles})
    return r


def _resolve_smiles(
    data: pd.DataFrame,
    batch_size: int,
    silent: bool,
    services: List[Service],
    db_path: str,
    update_db: bool = True,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    # Create list of compounds
    compounds = []
    for dortmund_id, row in data.iterrows():
        identifiers = []
        if row.get("cas_number"):
            identifiers.append(
                CompoundIdentifier(
                    identifier_type=CompoundIdentifierType.CAS_NUMBER,
                    value=row["cas_number"],
                )
            )
        if row.get("name"):
            identifiers.append(
                CompoundIdentifier(
                    identifier_type=CompoundIdentifierType.NAME, value=row["name"]
                )
            )
        if row.get("alternative_name"):
            identifiers.append(
                CompoundIdentifier(
                    identifier_type=CompoundIdentifierType.NAME,
                    value=row["alternative_name"],
                    details="alternative_name",
                )
            )
        identifiers.append(
            CompoundIdentifier(
                identifier_type=CompoundIdentifierType.CUSTOM,
                value=str(dortmund_id),
                details="id",
            )
        )
        compounds.append(Compound(identifiers=identifiers))

    # First try with local database
    resolver = CompoundResolver(
        services=[LocalDatabase(db_path=db_path, return_canonical_only=True)],
        silent=silent,
    )
    compound_identifiers_list_local = resolver.resolve(
        input_compounds=compounds,
        output_identifier_type=CompoundIdentifierType.SMILES,
        agreement=1,
        batch_size=batch_size,
        n_retries=1,
    )
    resolved = [
        _pura_results_to_dict(input_compound, output_identifiers)
        for input_compound, output_identifiers in compound_identifiers_list_local
        if output_identifiers is not None
    ]
    df = pd.DataFrame(resolved).set_index("id")
    unresolved_compounds = [
        compound
        for compound, identifiers in compound_identifiers_list_local
        if identifiers is None or len(identifiers) == 0
    ]
    # Resolve compounds with online services
    if len(unresolved_compounds) > 0:
        resolver = CompoundResolver(services=services, silent=silent)
        compound_identifiers_list = resolver.resolve(
            input_compounds=unresolved_compounds,
            output_identifier_type=CompoundIdentifierType.SMILES,
            agreement=2,
            batch_size=batch_size,
            n_retries=1,
        )
        resolved = [
            _pura_results_to_dict(input_compound, output_identifiers)
            for input_compound, output_identifiers in compound_identifiers_list
            if output_identifiers is not None
        ]
        if len(resolved) > 0:
            df_online = pd.DataFrame(resolved).set_index("id")
            df_online = df_online[~df_online["smiles"].isna()]
            # Update local database
            if update_db:
                logger.info("Updating local database")
                _update_pura_db(df_online, db_path)
            df = pd.concat([df, df_online])

    resolved = df[~df["smiles"].isna()]
    logger.info(
        f"{len(resolved)/len(data)*100:.0f}% resolved ({len(resolved)}/{len(data)})"
    )
    return df


def resolve_smiles(
    partitions: Dict[str, pd.DataFrame],
    db_path: str,
    batch_size: int = 5,
    update_db: bool = False,
) -> Dict[str, Callable[[], pd.DataFrame]]:
    """Resolve CAS numbers to SMILES strings

    Arguments
    ---------
    cas_number_partitions : dict
        A dictionary of pandas dataframes that represent partitions to cas numbers to lookup
    batch_size : int, optional
        Batch size for lookup. This should not be too large since lookups are asyncrhonous
        and can overload the services. Defaults to 5


    Notes
    -----
    This function uses a new package from the SRE group at Cambridge called pura.
    It uses the Chemical abstracts service, PubChem, the Chemical Identification Resolver
    from the NIH, and OPSIN from Cambridge.

    The underlying algorithm requires two of the services to resolve the same SMILES
    (after RDkit canonicalization).

    """
    # Prioritize cas number resolution first (cas/cir) then name (cir/opsin/pubchem)
    services = [CAS(), CIR(), Opsin(), PubChem()]
    return {
        key: partial(
            _resolve_smiles,
            data=partition,
            batch_size=batch_size,
            silent=True,
            services=services,
            db_path=db_path,
            update_db=update_db,
        )
        for key, partition in partitions.items()
    }


def check_atoms(mol, elements):
    for atom in mol.GetAtoms():
        if not atom.GetSymbol().upper() in elements:
            return False
    return True


def classify_molecules(molecules_df: pd.DataFrame, smiles_column: str = "smiles"):
    elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    statuses = [""] * len(molecules_df)
    for i, (id, row) in enumerate(molecules_df.iterrows()):
        smiles = row[smiles_column]
        if smiles is None or type(smiles) is int:
            continue
        # No ions
        if "+" in smiles or "-" in smiles:
            statuses[i] = "Exclude"
            continue
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if mol is None:
            statuses[i] = "Failed"
        elif check_atoms(mol, elements):
            statuses[i] = "Include"
        else:
            statuses[i] = "Exclude"
    molecules_df.loc[:, "status"] = statuses  # type: ignore
    return molecules_df


def merge_molecule_ids(
    smiles_data: pd.DataFrame, ids_to_cas: pd.DataFrame, id_column: str = "dortmund_id"
):
    """Merge dortmund ids with resolved SMILES"""
    ids_to_cas = ids_to_cas.set_index(id_column)
    return smiles_data[["smiles"]].join(ids_to_cas, how="left")


def merge_smiles(
    data: pd.DataFrame,
    smiles_lookup: pd.DataFrame,
    data_columns: Union[str, List[str]],
) -> pd.DataFrame:
    """Merge resolved SMILES into original dataframe"""
    if isinstance(data_columns, str):
        data_columns = [data_columns]

    smiles_lookup.index = smiles_lookup.index.astype(float)
    smiles_lookup.index = smiles_lookup.index.astype(int)
    for i, col in enumerate(data_columns):
        data = data.merge(smiles_lookup, how="left", left_on=col, right_index=True)
        data = data.rename(
            columns={col_l: f"{col_l}_{i+1}" for col_l in smiles_lookup.columns}
        )

    return data


def clean_resolve_crc_dipole_data(
    df: pd.DataFrame,
    pura_db_path: str,
    batch_size: int = 10,
    smiles_column: str = "smiles",
    dipole_moment_column: str = "dipole_moment",
    id_column: str = "id",
    filter_low_quality: bool = False,
):
    """Taken from pages 9-52 to 9-59 of CRC Handbook of Chemistry and Physics

    Data with ≈ or brackets is potentially low quality

    """
    # Clean data
    mu_col = dipole_moment_column
    dipoles = df["Dipole"].str.split("±", expand=True)
    dipoles = dipoles.rename(columns={0: mu_col, 1: "±dipole_moment"})
    approx = dipoles[mu_col].str.contains("≈")
    dipoles[mu_col] = dipoles[mu_col].str.replace("≈", "")
    brackets_extract = dipoles[mu_col].str.extract(r"(?<=\[)([\d+.]+)(?=\])")
    dipoles[mu_col] = brackets_extract[0].fillna(dipoles[mu_col])
    dipoles["low_quality"] = approx | brackets_extract[0].notnull()
    dipoles[mu_col] = dipoles[mu_col].astype(float)
    df = pd.concat([df, dipoles], axis=1)
    if filter_low_quality:
        df = df.loc[~df["low_quality"]]

    # Resolve SMILES
    results = resolve_identifiers(
        df["Name"].tolist(),
        output_identifier_type=CompoundIdentifierType.SMILES,
        services=[LocalDatabase(db_path=pura_db_path), CIR(), PubChem(), Opsin()],
        agreement=2,
        batch_size=batch_size,
        silent=True,
    )
    results = [
        (name, smiles[0])
        for name, smiles in results
        if (smiles is not None) and (len(smiles) > 0)
    ]
    results = pd.DataFrame(results, columns=["Name", smiles_column])
    df = df.merge(results, how="left", on="Name")
    df = df.dropna(subset=["smiles"]).reset_index(drop=False)
    df = df.rename(columns={"index": id_column})
    return df


def generate_rdkit_conformers(
    data: pd.DataFrame, smiles_column: str, id_column: str, batch_size: int
) -> Dict[str, Callable]:
    """Generate RDKit conformers for a dataframe of SMILES"""
    return parallel_runner(
        task=RDKitConformerRunner,  # type: ignore
        data=data,
        smiles_column=smiles_column,
        batch_size=batch_size,
        filter_kwargs={"id_column": id_column},
    )


def density_filtering(
    data: pd.DataFrame,
    density_bounds: Tuple[float, float],
    max_pressure: float = 150,
    density_column: str = "DEN",
    pressure_column: str = "P",
):
    return data[
        (
            (
                data[density_column].isna()
                | data[density_column].between(
                    density_bounds[0], density_bounds[1], inclusive="neither"
                )
            )
        )
        & ((data[density_column].notnull()) & (data[pressure_column] < max_pressure))
        | data[density_column].isna()
    ]


def get_wikipedia_dipole_moments() -> pd.DataFrame:
    """Get dipole moments from wikipedia"""
    # Site URL
    url = "https://en.wikipedia.org/wiki/Glossary_of_chemical_formulae"

    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text

    # Parse HTML code and get tables
    soup = BeautifulSoup(html_content, "lxml")
    tables = soup.find_all("table", attrs={"class": "wikitable sortable"})
    logger.info(f"Found {len(tables)} tables")

    # Get links from tables
    results = []
    try:
        for table in tqdm(tables):
            table_links = extract_table_links(table)
            for link in tqdm(table_links, position=1, leave=False):
                res = get_smiles_dipole_moment(link)
                results.append(res)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Stopping...")
        pass

    df = pd.DataFrame(results, columns=["smiles", "dipole_moment"])
    return df
