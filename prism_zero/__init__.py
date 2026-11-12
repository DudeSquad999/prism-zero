"""Offline candidate generation, constraint verification and local SFT."""
from importlib.metadata import PackageNotFoundError, version as _version
from .backends import LocalHFGenerator, build_messages, local_model_path
from .candidates import Candidate
from .dataset import (SCHEMA_VERSION, DatasetError, DatasetStore, dataset_fingerprint,
                      read_dataset, record_id, validate_record)
from .engine import PrismZeroEngine, Rule, RunResult
from .scripted import ScriptedModel
from .verifiers import ASTVerifier, JSONSchemaVerifier, MathEqualityVerifier, RegexVerifier

try:
    __version__ = _version('prism-zero')
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = '0.0.0+source'

__all__ = ["Candidate", "PrismZeroEngine", "Rule", "RunResult", "ASTVerifier",
           "JSONSchemaVerifier", "MathEqualityVerifier", "RegexVerifier",
           "LocalHFGenerator", "ScriptedModel", "build_messages", "local_model_path",
           "DatasetStore", "DatasetError", "SCHEMA_VERSION", "read_dataset",
           "validate_record", "record_id", "dataset_fingerprint", "__version__"]
