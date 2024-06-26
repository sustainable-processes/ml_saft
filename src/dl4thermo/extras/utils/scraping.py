"""Extract Data from Wikipedia"""
import logging
import re
from typing import Tuple, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup

wikipedia_base = "https://en.wikipedia.org"
logger = logging.getLogger(__name__)


def extract_table_links(table):
    body = table.find_all("tr")
    body_rows = body[1:]
    links = []
    for body_row in body_rows:
        synonym = body_row.find_all("td")[1]
        ahrefs = synonym.find_all("a")
        if len(ahrefs) == 0:
            continue
        rel_link = ahrefs[0].get("href")
        if rel_link:
            links.append(wikipedia_base + rel_link)
    return links


def get_smiles_dipole_moment(url: str) -> Tuple[Union[str, None], Union[str, None]]:
    """Get the dipole moment from a wikipedia chembox if it's there"""
    # Get the HTML and parse it
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")

    # Find chembox table
    chemboxes = soup.find_all("table", attrs={"class": "infobox ib-chembox"})
    if len(chemboxes) == 0:
        return None, None
    chembox = chemboxes[0]
    body = chembox.find_all("tr")
    body_rows = body[1:]

    # Get table entries
    rows = []
    for body_row in body_rows:
        row = [el.text.strip() for el in body_row.find_all("td")]
        rows.append(row)

    # Convert to dataframe and extract the values
    df = pd.DataFrame(data=rows)
    df = df.dropna(subset=[0])
    smiles_row = df[df[0].str.contains("SMILES")]
    mu_row = df[df[0] == "Dipole moment"]

    # Clean
    smiles: Union[str, None] = None
    if smiles_row.shape[0] > 0:
        smiles = smiles_row.iloc[0, 0].lstrip("SMILES\n")  # type: ignore
    mu = None
    if mu_row.shape[0] > 0:
        matches = re.match(r"^[\d.]+", mu_row[1].iloc[0])
        if matches:
            mu = matches[0]
    return smiles, mu
