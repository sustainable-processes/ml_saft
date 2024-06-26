import pathlib
import platform as p
import sqlite3
import warnings
from typing import List

import pandas as pd
from tqdm import tqdm

USE_ACCELERATION = True

try:
    from COSMOpy import CosmoPy, Mixture
    from COSMOpy.COSMOcompound import CompoundList, convertPropertyName, convertUnit
except ImportError:
    warnings.warn(
        (
            "COSMOpy not installed. If you are part of the Sustainable Reaction"
            "Engineering group Github organization, "
            "install with  pip install git+https://github.com/sustainable-processes/COSMOpy.git`"
            "Otherwise, contact Biovia to get a copy of COSMOpy."
        )
    )


def guess_cosmobase_path():
    system = p.system()
    if system == "Darwin":
        base_path = pathlib.Path("/Applications/BIOVIA")
    elif system == "Linux":
        base_path = pathlib.Path.home() / "BIOVIA"
    elif system == "Windows":
        warnings.warn("This code has not been tested on Windows.")
        base_path = pathlib.Path("C:/Program Files/BIOVIA")
    else:
        raise ValueError(f"System {system} not supported")

    path = (
        base_path
        / "COSMOtherm2020/COSMOtherm/DATABASE-COSMO/COSMObase2020/BP-TZVPD-FINE.db"
    )
    return path


def get_cosmobase_df(path=None):
    """Get COSMO search df

    path is the path to the database
    """

    if path is None:
        path = guess_cosmobase_path()
    db_path = pathlib.Path(path)

    # Connect to SQLite3
    conn = sqlite3.connect(db_path)

    # Read into dataframe
    df = pd.read_sql(
        "SELECT uniqueCode12,smiles,casNumber,compoundID,compoundName FROM SearchResultSet",
        conn,
    )

    # Close SQLite3 connection
    conn.close()

    return df


class CosmoCalculate:
    CAS = "CAS"
    UNICODE = "UNICODE"
    NAME = "NAME"
    SMILES = "SMILES"

    """  A convencience wrapper around CosmoPy
    
    Parameters
    ---------- 
    calculation_name : str
        Name of the calculation. For example "Activity Coefficient".
    lookup_name : str
        Name of the column for molecule identifiers in the Pandas series 
        passed in at call time. Defaults to "casNumber".
    lookup_type : str, optional
        Identifier used for looking up molecules. Defaults to CosmoCalculate.CAS but
        can also be UNICODE, NAME, or SMILES.
    level : str, optional
        Level for running calculations. Defaults to TZVPD-FINE.
    n_cores : int, optional
        Number of cores to use for calculations in total. Defaults to 1.
    cores_per_job : int, optional
        Number of cores to use for each job. Defaults to 1.
    
    
    Examples
    --------
    >>> calc_func = CosmoCalculate("Boiling Point",  \
                                   "Tboil", \
                                   lookup_name="uniqueCode12", \
                                   lookup_type=CosmoCalculate.UNICODE, \
                                   background=True, \
                                   n_cores=1)
    >>> df = get_cosmo_search_df()
    >>> mols = [calc_func(row) for _, row in  \
                df.iloc[rows_read:rows_read + batch_size].iterrows()]
    
    """

    def __init__(self, calculation_names: List[str], **kwargs):
        self.calculation_names = calculation_names
        self.lookup_name = kwargs.get("lookup_name", "casNumber")
        self.lookup_type = kwargs.get("lookup_type", self.CAS)

        self.search_df = get_cosmobase_df()

        # Set up cosmo-rs
        self.ct = CosmoPy().cosmotherm
        self.ct.setLevel(kwargs.get("level", "TZVPD-FINE"))
        n_cores = kwargs.get("n_cores", 1)
        self.ct.setUseCores(kwargs.get("cores_per_job", 1))
        if n_cores == "max":
            n_cores = self.ct.getFreeCores()
        self.ct.setNCores(int(n_cores))

        # Create scracth directory if it doesn't already exist
        pathlib.Path("scratch/").mkdir(exist_ok=True)
        self.ct.setScratch("scratch/")

        # Calculate in background. Need to manage queue
        self.background = kwargs.get("background", False)

    def __call__(self, *rows, **kwargs):
        """Calculate properties using COSMOtherm

        Parameters
        ----------
        rows : pd.Series
            Row of pandas dataframe that has a casNumber column
        xmin : float
            Minimum of composition grid for binary mixtures
        xmax : float
            Maximum of composition grid for binary mixtures
        xstep : float
            Step of composition grid for binary mixtures
        Tmin : float
            Minimum of temperature grid for binary mixtures
        Tmax : float
            Maximum of temperature grid for binary mixtures
        Tstep : float
            Step of temperature grid for binary mixtures

        Returns
        -------
        Row with property value inserted

        """
        mols = []
        for row in rows:
            value = row[self.lookup_name]
            mol = self.lookup_molecule(value)
            if mol is None:
                raise ValueError(f"No molecule found for {value}")
            mols.append(mol)
        # Run calculations for a single molecule
        self.ct.cmdline = ""
        if len(mols) == 1:
            self.ct.compound = mols[0]
            for calculation_name in self.calculation_names:
                self.ct.append(method=calculation_name, **kwargs)
                # self.ct.define(mols[0], calculation_name, **kwargs)
            self.ct.compute(use_scratch=True, background=self.background)

        # Run calculation for mixture
        else:
            # Create mixture
            mixture_name = "".join(
                [f"{row[self.lookup_name]}__" for row in rows]
            ).rstrip("__")
            mixture = Mixture(mixture_name)
            for mol in mols:
                mixture.addCompound(mol)

            # Define calculation
            binary_check_1 = (
                kwargs.get("xmin") or kwargs.get("xlist") or kwargs.get("Tmin")
            )
            binary_check_2 = len(mols) == 2
            if binary_check_1 and binary_check_2:
                for calculation_name in self.calculation_names:
                    self.ct.defineBinaryGrid(
                        mixture=mixture, method=calculation_name, **kwargs
                    )
            # Simple mixture calculation
            else:
                for calculation_name in self.calculation_names:
                    self.ct.define(mixture, calculation_name, **kwargs)

            # Schedule comptuation
            self.ct.compute(use_scratch=True, background=self.background)

        return self.ct.file_name

    def lookup_molecule(self, value):
        if self.lookup_type == self.CAS:
            row = self.search_df[self.search_df["casNumber"] == value]
            if not row.empty:
                value = str(row["uniqueCode12"].values[0])
                mol = self.ct.searchUnicode(value)[0]  # type: ignore
            else:
                mol = None
        elif self.lookup_type == self.UNICODE:
            mol = self.ct.searchUnicode(value)
            if isinstance(mol, list) and len(mol) >= 1:
                mol = mol[0]
            else:
                mol = None
        elif self.lookup_type == self.NAME:
            row = self.search_df[self.search_df["compoundName"] == value]
            value = str(row["uniqueCode12"].values[0])
            mol = self.ct.searchUnicode(value)[0]  # type: ignore
        elif self.lookup_type == self.SMILES:
            row = self.search_df[self.search_df["smiles"] == value]
            value = str(row["uniqueCode12"].values[0])
            mol = self.ct.searchUnicode(value)[0]  # type: ignore
        else:
            raise ValueError(f"Unknown lookup type: {self.lookup_type}.")
        return mol

    def read(self, path):
        compound_list = CompoundList()

        # Initialize name and compound list and get output file (.out)
        tab = self.ct.getResultPath(path=path)
        out = tab[:-3] + "out"

        self.readCosmotherm(out, compound_list)
        for calculation_name in self.calculation_names:
            self.readTable(calculation_name, tab, compound_list)
        return compound_list

    def readCosmotherm(self, path, compound_list):
        """Copied from COSMOpy with minor modifications"""

        # Initialize reading
        compound = None
        compound2 = None
        title = ""
        check = not USE_ACCELERATION
        is_compound_information = False

        # Open file
        try:
            out_file = open(path, "rb")
        except:
            raise IOError("ERROR: File %s could not be opened.\n" % (path))

        # Read lines from table file
        try:
            for line in out_file:
                line = bytes.decode(line)
                line = line.strip()
                # Empty line
                if line == "":
                    is_license_information = False
                    if not is_compound_information:
                        compound = None
                        compound2 = None

                elif line.startswith("---"):
                    is_compound_information = False
                    continue

                elif line.startswith("COSMOtherm Version"):
                    self.version = line[18:].strip()

                elif line.startswith("BIOVIA COSMOtherm Version"):
                    self.version = line[
                        25:
                    ]  # "BIOVIA COSMOtherm Version 2019 (build: 20.0.0 Revision 5562)" --> "2019 (build: 20.0.0 Revision 5562)"

                # VAP and COSMO file information lines
                elif line.startswith("Reading COSMO information"):
                    # Example line: "Reading COSMO information for molecule 1 from file C:\Program Files\COSMOlogic\COSMOthermX17\COSMOtherm\DATABASE-COSMO\BP-TZVPD-FINE\1/1-octanol_c0.cosmo"
                    query = "from file "
                    pos1 = line.find(query) + len(query)
                    cosmo_path = line[pos1:]
                    compound2 = compound_list.getCompoundCosmo(cosmo_path, add_new=True)
                    title = line
                    is_compound_information = False

                # Title
                elif line.startswith("Compound Information for compound"):
                    # Example line: "Compound Information for compound 1 (COSMO file C:\Program Files\COSMOlogic\COSMOthermX14\COSMOtherm\DATABASE-COSMO\BP-TZVPD-FINE\h/hexane_c0.cosmo)"
                    query = "COSMO file "
                    pos1 = line.find(query) + len(query)
                    pos2 = -1
                    cosmo_path = line[pos1:pos2]

                    compound = compound_list.getCompoundCosmo(cosmo_path, add_new=True)
                    compound2 = compound
                    title = line
                    is_compound_information = True

                # Table
                elif line.startswith("START job"):
                    is_compound_information = False
                    break  # COSMOtherm specific calculations not read here, use method readTable() instead

                elif (compound2 is not None) and (line.startswith("Molecule ")):
                    value_list = line.split()
                    if value_list[-3] == "N**2":
                        # Refractive index
                        value = float(value_list[-1])
                        property_name = "N**2"
                        unit = "1"
                        compound2.setProperty(  # type: ignore
                            property_name, value, unit=unit, title=title, check=check
                        )  # type: ignore
                    elif value_list[-3] == "EPSILON":
                        value = float(value_list[-1])
                        property_name = "epsilon"
                        unit = "1"
                        compound2.setProperty(  # type: ignore
                            property_name, value, unit=unit, title=title, check=check
                        )  # type: ignore
                    elif "I(1,2,3)" in value_list:
                        moment_inertia_list = [
                            "PrincipalMomentInertia(1)",
                            "PrincipalMomentInertia(2)",
                            "PrincipalMomentInertia(3)",
                        ]
                        unit = "1"
                        for idx_value, value in enumerate(value_list[-3:]):
                            value = float(value)
                            property_name = moment_inertia_list[idx_value]
                            compound2.setProperty(  # type: ignore
                                property_name,
                                value,
                                unit=unit,
                                title=title,
                                check=check,
                            )

                elif (compound2 is not None) and (
                    line.startswith("Symmetry point group ")
                ):
                    value = line[21:24].strip()
                    if "gas phase" in line:
                        property_name = "SymmetryPointGroup(Gas)"
                    else:
                        property_name = "SymmetryPointGroup(COSMO)"
                    unit = "1"
                    compound2.setProperty(  # type: ignore
                        property_name, value, unit=unit, title=title, check=check
                    )

                elif (compound is not None) and (":" in line):
                    value_list = line[24:].split()
                    parameter = line[:23].strip()
                    parameter_list = [
                        "Atomic weights",
                        "E_COSMO+dE",
                        "E_gas",
                        "E_COSMO-E_gas+dE",
                        "E_diel",
                        "Averaging corr dE",
                        "EvdW in continuum",
                        "Area",
                        "Volume",
                        "Molecular Weight",
                        "Total COSMO charge",
                        "H-bond moment (accept)",
                        "H-bond moment (donor)",
                    ]
                    replace_list = [
                        "AtomicWeights",
                        "E_cosmo",
                        "E_gas",
                        "E_solv",
                        "E_diel",
                        "av_dE",
                        "E_vdW",
                        "Area",
                        "Vcosmo",
                        "MolWeight",
                        "Qcosmo",
                        "HB_acc3",
                        "HB_don3",
                    ]
                    if parameter in parameter_list:
                        if ("Atomic weights") in parameter and value_list[0].endswith(
                            "..."
                        ):
                            value_list[0] = value_list[0][
                                :-3
                            ]  # quick fix for abreviated weights
                        value = float(value_list[0])
                        if len(value_list) > 1:
                            unit = value_list[1]
                        else:
                            unit = "1"
                        property_name = replace_list[parameter_list.index(parameter)]
                        compound.setProperty(  # type: ignore
                            property_name, value, unit=unit, title=title, check=check
                        )
                    elif parameter == "Dipole moment (t,x,y,z)":
                        dipole_list = ["Dipole", "DipoleX", "DipoleY", "DipoleZ"]
                        unit = value_list[4]
                        for idx_value, value_split in enumerate(value_list[:-1]):
                            value = float(value_split)
                            property_name = dipole_list[idx_value]
                            compound.setProperty(  # type: ignore
                                property_name,
                                value,
                                unit=unit,
                                title=title,
                                check=check,
                            )
                    elif parameter == "Sigma moments (1-6)":
                        unit = "1"
                        for idx_value, value_split in enumerate(value_list):
                            value = float(value_split)
                            property_name = "sig%i" % (idx_value + 1)
                            compound.setProperty(  # type: ignore
                                property_name,
                                value,
                                unit=unit,
                                title=title,
                                check=check,
                            )
                    elif parameter == "H-bond moment (accept)":
                        unit = "1"
                        for idx_value, value_split in enumerate(value_list):
                            value = float(value_split)
                            property_name = "HB_acc%i" % (idx_value + 1)
                            compound.setProperty(  # type: ignore
                                property_name,
                                value,
                                unit=unit,
                                title=title,
                                check=check,
                            )
                    elif parameter == "H-bond moment (donor)":
                        unit = "1"
                        for idx_value, value_split in enumerate(value_list):
                            value = float(value_split)
                            property_name = "HB_don%i" % (idx_value + 1)

                            compound.setProperty(  # type: ignore
                                property_name,
                                value,
                                unit=unit,
                                title=title,
                                check=check,
                            )
            out_file.close()
        except:
            try:
                out_file.close()
            finally:
                raise IOError("ERROR: Reading properties from file %s.\n" % (path))

        return compound_list

    def readTable(self, input_name: str, path: str, compound_list):
        """Copied from COSMOpy with minor modifications"""

        # Open file
        try:
            tabfile = open(path, "rb")
        except:
            raise IOError("ERROR: File %s could not be opened.\n" % (path))

        # Table handling
        contentTitle = 0
        contentHeader = 1
        contentTable = 2
        contentTableLLE = 3
        contentNr = contentTable

        # Initialize
        title = ""
        mixture = None
        lle_found = False
        is_format_15 = True
        table_header = list()
        check = not USE_ACCELERATION

        # Sigma-Profiles (not yet implemented for version 16)
        contentSigmaProfile = 4
        contentSigmaCompounds = 5
        contentSigmaPotential = 6
        contentSigmaMixtures = 7
        sigma_profiles = list()
        sigma_compounds = list()
        relative_fractions = []
        use_weight_fraction = False
        temperature = None
        T_unit = ""

        # Read lines from table file
        try:
            prev_line = ""
            for line in tabfile:
                line = bytes.decode(line)
                line = line.strip()

                # Empty line: Switch between "Title" -> "Table header" -> "Table"
                if line.startswith("Property  job"):
                    contentNr = contentTitle

                if line == "":
                    if prev_line == "":
                        contentNr = contentTitle

                    elif contentNr == contentTableLLE:
                        mixture.setProperty(  # type: ignore
                            property_name="LLE",
                            value=lle_found,
                            unit="BOOLEAN",
                            title=title,
                            check=check,
                        )
                        contentNr = contentTitle

                    elif contentNr == contentTable:
                        contentNr = contentTitle
                    elif contentNr == contentTitle:
                        contentNr = contentHeader
                    elif contentNr == contentSigmaProfile:
                        contentNr = contentSigmaCompounds
                    elif contentNr == contentSigmaPotential:
                        contentNr = contentSigmaMixtures
                    elif contentNr == contentSigmaCompounds:
                        contentNr = contentHeader
                    elif contentNr == contentSigmaMixtures:
                        contentNr = contentHeader

                # Property title
                elif contentNr == contentTitle:
                    # New property: "Property  job 1 : Vapor pressures ;"
                    if line.startswith("Property  job"):
                        pos_start = line.index(":") + 2
                        # For LLE and NRTL, use previously defined mixture
                        if not (("LLE results" in line) or ("NRTL" in line)):
                            mixture = None
                            title = (
                                line + " - composition " + input_name
                            )  # Store full line for job number
                        else:
                            # Mixture from previous calculation
                            title += " ; " + line[pos_start:]
                        temperature = None
                        T_unit = "K"
                        lle_found = False

                    # Compound or mixture: "Compounds job 2 : diethylether (1) ; 1-octanol (2) ;"
                    if line.startswith("Compounds job"):
                        pos_start = line.index(":") + 2
                        title += " ; " + line[pos_start:]
                        entries = line[pos_start:].split(";")
                        mixture = Mixture()
                        for entry in entries:
                            entry = entry.strip()
                            if entry:
                                name = " ".join(entry.split()[:-1]).strip()
                                compound = compound_list.getCompound(name, add_new=True)
                                mixture.addCompound(compound)
                                relative_fractions.append(
                                    0.0
                                )  # Molar or weight fractions added later on...

                        # Search or append compound or mixture to compound list

                        # Single compound
                        if len(mixture.compounds) == 1:
                            compound = mixture.compounds[0]
                            mixture = compound_list.getCompound(compound, add_new=True)

                        # Mixture composition
                        else:
                            mixname = mixture.getComponentString()
                            if mixname:
                                mixture.setName(
                                    mixname
                                )  # Mixture has classical, human readable name
                                mixture_search = compound_list.getCompound(mixname)
                                if mixture_search is None:
                                    # Check for previous mixtures in compound list
                                    isfound = False
                                    for mixture_search in compound_list.compounds:
                                        if isinstance(mixture_search, Mixture):
                                            if (
                                                mixture_search.getComponentString().lower()
                                                == mixture.getComponentString().lower()
                                            ):
                                                isfound = True
                                                break
                                    if not isfound:
                                        mixture_search = None
                                        compound_list.addMixture(mixture)
                                if mixture_search is not None:
                                    mixture = mixture_search

                    # Settings and composition: "Settings  job 1 : T= 298.15 K ; x(1)= 0.2889 x(2)= 0.7111 ;"
                    elif line.startswith("Settings  job"):
                        pos_start = line.index(":") + 2
                        title += " ; " + line[pos_start:]
                        entries = line[pos_start:].split(";")
                        for entry in entries:
                            entry = entry.strip()
                            if entry:

                                # Temperature "T= 298.15 K"
                                if entry[0] == "T":
                                    T_split = entry.split()
                                    temperature = float(T_split[1])
                                    T_unit = T_split[2][-1]
                                    if T_unit != "K":
                                        temperature = convertUnit(
                                            temperature, T_unit, "K", verbose=False
                                        )

                                # Composition "x(1)= 0.2889 x(2)= 0.7111"
                                elif entry[0] in ["x", "c"]:
                                    if entry[0] == "c":
                                        use_weight_fraction = True
                                    else:
                                        use_weight_fraction = False
                                    x_split = entry.split()
                                    for idx_entry, x_entry in enumerate(x_split):
                                        x_entry = x_entry.strip()
                                        idx_compound = 0
                                        if idx_entry % 2 == 0:
                                            # "x(1)="
                                            idx_compound = int(x_entry[2:-2]) - 1
                                            while idx_compound >= len(
                                                relative_fractions
                                            ):
                                                relative_fractions.append(
                                                    0.0
                                                )  # Molar or weight fractions added later on...
                                        else:
                                            # "0.2889"
                                            relative_fractions[idx_compound] = float(
                                                x_entry
                                            )

                                # Phases and composition in first phase  "Phase 1: x(1)= 0.3248 x(2)= 0.6752 ; Phase 2: x(2)= 1.0000 ;" - composition of phase 2 ignored for the time being
                                elif entry.startswith("Phase 1:"):
                                    entry = entry[9:]
                                    if entry[0] == "c":
                                        use_weight_fraction = True
                                    else:
                                        use_weight_fraction = False
                                    x_split = entry.split()
                                    for idx_entry, x_entry in enumerate(x_split):
                                        x_entry = x_entry.strip()
                                        idx_compound = 0
                                        if idx_entry % 2 == 0:
                                            # "x(1)="

                                            idx_compound = int(x_entry[2:-2]) - 1
                                            while idx_compound >= len(
                                                relative_fractions
                                            ):
                                                relative_fractions.append(
                                                    0.0
                                                )  # Molar or weight fractions added later on...
                                        else:
                                            # "0.2889"
                                            relative_fractions[idx_compound] = float(
                                                x_entry
                                            )

                    # Units: "Units     job 1 : Energies in kcal/mol ; Volume in A^3 ; Temperature in K ;"
                    elif line.startswith("Units     job"):
                        pos_start = line.index(":") + 2

                        # Temperature not in K (set per default)
                        if "Temperature in C" in title:
                            T_unit = "C"
                        elif "Temperature in F" in title:
                            T_unit = "F"

                        # Retain backwards compatibility, as defined by _getUnitFromTitle
                        unit_line = line[pos_start:].replace(" ;", " -")
                        unit_line = unit_line.replace(
                            "Henry constant in", "Henry constant is in"
                        )
                        unit_line = unit_line.replace("Energies in", "energies are in")
                        unit_line = unit_line.replace("Pressure in", "pressure is in")
                        unit_line = unit_line.replace("Volume in", "volume is in")
                        unit_line = unit_line.replace(
                            "Temperature in", "temperature is in"
                        )
                        unit_line = unit_line.replace("Area in", "area is in")
                        unit_line = unit_line.replace("Density in", "density is in")
                        unit_line = unit_line.replace("Viscosity in", "viscosity is in")
                        title += " ; " + unit_line + " ;"

                    # General: "General   job 1 : Mass solubility w_solub = x_s*MW(s)/((1-x_s)*MW(solvent)) is in g/g ; Molar solubility S is in mol/l (solution) ; Solvent density =  793.008 g/l ;"
                    elif line.startswith("General   job"):
                        pos_start = line.index(":") + 2

                        # Retain backwards compatibility
                        unit_line = line[pos_start:].replace(" ;", " -")
                        title += " ; " + unit_line + " ;"

                    elif line.startswith("LLE point found"):
                        # "LLE point found at x`(1) = 0.00011281  x`(2) = 0.99988719 and x``(1) = 0.81731313  x``(2) = 0.18268687  (T =  298.15 K) - ptot =      31.8367634 mbar  y(1) =  0.00228506 y(2) =  0.99771494 - job 1"
                        lle_found = True  # LLE points not investigated further, but LLE line stored here
                        mixture.setProperty(  # type: ignore
                            property_name="LLE",
                            value=lle_found,
                            unit="BOOLEAN",
                            title=title,
                            check=check,
                        )
                        entries = line.split()
                        for idx_entry, entry in enumerate(entries):
                            if entry.startswith("x`"):
                                property_name = "LLE " + entry
                                value = entries[idx_entry + 2]
                                mixture.setProperty(  # type: ignore
                                    property_name,
                                    value,
                                    unit="1",
                                    title=title,
                                    check=check,
                                )

                    elif line.startswith("No LLE found"):
                        lle_found = False
                        mixture.setProperty(  # type: ignore
                            property_name="LLE",
                            value=lle_found,
                            unit="BOOLEAN",
                            title=title,
                            check=check,
                        )

                # Table header: "T           PVtot         mu(Liquid)            mu(Gas)          H(Vapori)"
                elif contentNr == contentHeader:
                    table_header = line.split()
                    contentNr = contentTable

                # Table values of particular row
                elif contentNr in [contentTable, contentTableLLE]:
                    table_columns = line.split()

                    # No mixture composition defined
                    # if not relative_fractions:
                    #     relative_fractions = None

                    # Pure compound or mixture
                    compound = mixture
                    if (isinstance(compound, Mixture)) and (
                        relative_fractions is not None
                    ):
                        if (1.00 in relative_fractions) and (mixture is not None):
                            idx_compound = relative_fractions.index(1.00)
                            if len(mixture.compounds) > idx_compound:
                                compound = mixture.compounds[idx_compound]

                    # Search results table
                    for idx_column, property_name in enumerate(table_header):

                        # Similarity
                        if property_name.isdigit():
                            if "Sigma match similarity" in title:
                                property_name = "SimilaritySigma(%s)" % (property_name)
                            elif "Sigma potential similarity" in title:
                                property_name = "SimilarityPotential(%s)" % (
                                    property_name
                                )

                        # General internal property name
                        property_internal = convertPropertyName(property_name)

                        # Number
                        if property_internal in ["NR"]:
                            continue  # Don't enter number into property tables

                        # LLE: "Property  job 1 : LLE results for multinary system ;"
                        if "LLE results" in title:
                            contentNr = contentTableLLE
                            # if lle_found:
                            #    break  # LLE points not investigated further, move to next line in LLE table!
                            if property_internal.startswith(
                                "XD"
                            ):  # As in "x`(1)     x`(2)     x`(3) ..."
                                value = table_columns[idx_column]
                                if float(value) > 1e-7:
                                    lle_found = True
                                    mixture.setProperty(  # type: ignore
                                        property_name="LLE " + property_name,
                                        value=value,
                                        unit="1",
                                        T=temperature,
                                        title=title,
                                        c=relative_fractions,
                                    )
                                    # break  # LLE points not investigated further, move to next line in LLE table!
                            else:
                                break  # LLE points not investigated further, move to next line in LLE table!

                        # NRTL parameters (introduced in 08/19: general activity coefficient models)
                        if (
                            property_internal == "MODEL"
                        ):  # and (table_columns[idx_column] == "NRTL"):
                            if (
                                "model parameters for the activity coefficients"
                                in title
                            ):
                                property_name = "%s(%s)" % (
                                    table_columns[idx_column + 1],
                                    table_columns[idx_column],
                                )  # Example: "Tau21(NRTL)"
                                value = table_columns[idx_column + 2]
                                # property_name = table_columns[idx_column + 1]
                                # value = None
                                # if (property_name in ["Alpha", "Alpha21", "Tau12", "Tau21", "rms"]):
                                #    property_name = property_name + "(NRTL)"
                                #    value = table_columns[idx_column + 2]
                                if value is not None:
                                    if use_weight_fraction:
                                        mixture.setProperty(  # type: ignore
                                            property_name=property_name,
                                            value=value,
                                            unit="1",
                                            T=temperature,
                                            title=title,
                                            c=relative_fractions,
                                        )
                                    else:
                                        mixture.setProperty(  # type: ignore
                                            property_name=property_name,
                                            value=value,
                                            unit="1",
                                            T=temperature,
                                            title=title,
                                            x=relative_fractions,
                                        )
                                break  # No more values in table, move to next line in NRTL table!

                        # Compound name
                        if property_internal in [
                            "COMPOUND",
                            "MOLECULECONFORMER",
                            "ATMOSCOMP",
                            "SOLUTE",
                            "COMPONENT",
                        ]:
                            name = table_columns[idx_column]
                            compound = compound_list.getCompound(name, add_new=True)
                            continue  # Don't enter compound name into property tables

                        # Molar composition
                        elif property_internal in [
                            "X1",
                            "X2",
                            "X3",
                            "X4",
                            "X5",
                            "X6",
                            "X7",
                            "X8",
                            "X9",
                        ]:
                            use_weight_fraction = False
                            if property_internal == "X1":
                                relative_fractions = list()
                            relative_fractions.append(float(table_columns[idx_column]))

                        # Temperature
                        elif property_internal in ["T"]:
                            if "K" not in T_unit:
                                temperature = convertUnit(
                                    float(table_columns[idx_column]), T_unit, "K"
                                )
                            else:
                                temperature = float(table_columns[idx_column])
                            continue  # Don't enter temperature into property tables

                        # Compound or mixture value - Add to property store
                        elif compound is not None:
                            if idx_column < len(table_columns):
                                value = table_columns[idx_column]
                                unit = self._getUnitFromTitle(property_name, title)

                                if use_weight_fraction:
                                    compound.setProperty(
                                        property_name=property_name,
                                        value=value,
                                        unit=unit,
                                        T=temperature,
                                        title=title,
                                        c=relative_fractions,
                                        check=check,
                                    )
                                else:
                                    compound.setProperty(
                                        property_name=property_name,
                                        value=value,
                                        unit=unit,
                                        T=temperature,
                                        title=title,
                                        x=relative_fractions,
                                        check=check,
                                    )

                                # Special property boiling point
                                if temperature is not None:
                                    if property_internal == "PVTOT":
                                        try:
                                            value = float(value)
                                            pvap = convertUnit(value, unit, "kPa")
                                            if (pvap > 101.0) and (
                                                pvap < 101.5
                                            ):  # = 1 atm (+/-) 0.5 kPa
                                                compound.setProperty(
                                                    property_name="Tboil",
                                                    value=temperature,
                                                    unit="K",
                                                    title=title,
                                                    check=check,
                                                )
                                        except:
                                            pass
                                    if property_internal == "PVSAT":
                                        if "Flash point temperature" in title:
                                            compound.setProperty(
                                                property_name="T_flash",
                                                value=temperature,
                                                unit="K",
                                                title=title,
                                                check=check,
                                            )

                prev_line = line

            # Closing file handling
            if contentNr == contentTableLLE:
                mixture.setProperty(  # type: ignore
                    property_name="LLE",
                    value=lle_found,
                    unit="BOOLEAN",
                    title=title,
                    check=check,
                )

            tabfile.close()

        # Error Handling
        except Exception as e:
            try:
                tabfile.close()
            finally:
                raise IOError(
                    "ERROR: Reading properties from file %s (%s).\n" % (path, e)
                )

        return compound_list

    def _getUnitFromString(self, title, part):
        if title:
            title_lower = title.lower()
        else:
            title_lower = ""
        if part:
            part_lower = part.lower()
        else:
            part_lower = ""
        if part_lower in title_lower:
            left_index = title_lower.index(part_lower) + len(part_lower)
            right_index = title.find(" - ", left_index) + 1
            unit = title[left_index:right_index].strip()
        else:
            unit = "1"
        return unit

    def _getUnitFromTitle(self, property_name, title):

        property_name = convertPropertyName(property_name)
        if property_name == "H":
            # This order - first Henry constants, then energies - is necessarily required!
            unit = self._getUnitFromString(title, "Henry constant is in")
            if unit == "1":
                unit = self._getUnitFromString(title, "energies are in")
        elif property_name in ["HENRYCONST", "LOGHENRYCONST"]:
            unit = self._getUnitFromString(title, "Henry constant is in")
        elif property_name in [
            "PV",
            "LOGPVTOT",
            "LOGPVEXP",
            "PVTOT",
            "PVEXP",
            "PTOT",
            "PVSAT",
            "PCRIT",
        ]:
            unit = self._getUnitFromString(title, "pressure is in")
        elif property_name in ["VOLUME", "VCOSMO", "VCRIT", "EXPVOLUME", "COSMOVOLUME"]:
            unit = self._getUnitFromString(title, "volume is in")
        elif property_name in ["MOLWEIGHT"]:
            unit = "g/mol"
        elif property_name in ["WSOLUB"]:
            unit = self._getUnitFromString(
                title, "mass solubility w_solub = x_s*MW(s)/((1-x_s)*MW(solvent)) is in"
            )
        elif property_name in ["WSOLUBMGL"]:
            unit = "mg/l"
        elif property_name in ["LOG10S"]:
            unit = self._getUnitFromString(title, "molar solubility S is in")
        elif property_name in ["TMELT", "T", "TBOIL", "TCRIT"]:
            unit = self._getUnitFromString(title, "temperature is in")
        elif property_name in [
            "ECOSMODEMU",
            "HINT",
            "HMF",
            "HHB",
            "HVDW",
            "ERING",
            "ECOSMODE",
            "EGAS",
            "ECOSDEEGAS",
            "EDIEL",
            "DE",
            "MULIQUID",
            "MUGAS",
            "MU",
            "HVAPORI",
            "DGFUS",
            "HE",
            "GE",
            "MU1RTLNX1",
            "MU2RTLNX2",
            "MU3RTLNX3",
            "TS",
            "GSOLV",
            "GPVEXP",
            "HPVEXP",
            "MUSELF",
            "MUSOLV",
            "MUWATER",
            "HSUBLIM",
            "HTCORR",
        ]:
            unit = self._getUnitFromString(title, "energies are in")
        elif property_name in ["LOG10P"]:
            unit = self._getUnitFromString(title, "pressure is in")
        elif property_name in ["AREA"]:
            unit = self._getUnitFromString(title, "area is in")
        elif property_name in ["OVERALLKOHCM3S"]:
            unit = "cm^3/molecules/s"
        elif property_name in ["HALFLIFEDAYS"]:
            unit = "days"
        elif property_name in ["DENSITY", "EXPDENSITY"]:
            unit = self._getUnitFromString(title, "density is in")
        elif property_name in ["VISCOSITY"]:
            unit = self._getUnitFromString(title, "viscosity is in")
        else:
            unit = "1"

        if property_name.startswith("LOG10"):
            unit = "log(" + unit + ")"
        elif property_name.startswith("LOG"):
            unit = "log(" + unit + ")"
        elif property_name.startswith("LN"):
            unit = "ln(" + unit + ")"
        unit = unit.replace("**", "^")

        return unit
