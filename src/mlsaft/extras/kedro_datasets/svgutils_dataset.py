import io
from typing import Any, Dict, Literal

from kedro.io import AbstractDataSet
from lxml import etree
from svgutils.compose import Figure


class SvgUtilsDataset(AbstractDataSet):
    def __init__(self, filepath: str, format: Literal["svg", "png", "pdf"] = "svg"):
        """Saves a svgutils.compose.Figure object to a file.

        Args:
            filepath: The location of the image file to load / save data.
            format: The format to save the file in. Can be "svg", "png", or "pdf".
        """
        self._filepath = filepath
        self._format = format

    def _load(self):
        raise NotImplementedError("Loading not supported")

    def _save(self, figure: Figure):
        if self._format == "svg":
            figure.save(self._filepath)
        else:
            import cairosvg

            out = etree.tostring(
                figure.root,
                xml_declaration=True,  # type: ignore
                standalone=True,  # type: ignore
                pretty_print=True,  # type: ignore
                encoding=None,  # type: ignore
            )
            b = io.BytesIO(out)
            if self._format == "png":
                cairosvg.svg2png(bytestring=b.read(), write_to=self._filepath)
            elif self._format == "pdf":
                cairosvg.svg2pdf(bytestring=b.read(), write_to=self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, format=self._format)
