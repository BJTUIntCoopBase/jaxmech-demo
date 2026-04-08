"""jaxmech.io.abaqus - ABAQUS INP I/O utilities (demo subset)."""

from jaxmech.io.abaqus.inp_metadata import (
    parse_inp_metadata,
    parse_inp_metadata_batch,
)
from jaxmech.io.abaqus.inp_metadata_store import (
    analysis_input_sidecar_path,
    delete_analysis_input_sidecar,
    decode_analysis_input_bundle,
    load_analysis_input_sidecar_payload,
    save_analysis_input_snapshot,
)
from jaxmech.io.abaqus.solid_inp import (
    load_parsed_model_mat,
    parse_inp_model,
    save_parsed_model_mat,
)

__all__ = [
    "delete_analysis_input_sidecar",
    "decode_analysis_input_bundle",
    "load_parsed_model_mat",
    "load_analysis_input_sidecar_payload",
    "parse_inp_metadata",
    "parse_inp_metadata_batch",
    "parse_inp_model",
    "analysis_input_sidecar_path",
    "save_analysis_input_snapshot",
    "save_parsed_model_mat",
]
