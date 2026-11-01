"""Regenerate the in-memory ML Program test fixture with coremltools 9.

Run from the repository root with Python, coremltools and numpy installed.
The generated Rust fixture has no runtime Python dependency.
"""
import pathlib
import tempfile

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


@mb.program(input_specs=[mb.TensorSpec((1, 16), dtype=types.fp16)],
            opset_version=ct.target.iOS18)
def weighted_model(x):
    return mb.matmul(x=x, y=2 * np.eye(16, dtype=np.float16), name="y")


model = ct.convert(weighted_model, minimum_deployment_target=ct.target.iOS18,
                   skip_model_load=True)
spec = model.get_spec()
spec.description.metadata.userDefined.pop("com.github.apple.coremltools.conversion_date", None)
specification = spec.SerializeToString(deterministic=True)

with tempfile.TemporaryDirectory() as temporary:
    package = pathlib.Path(temporary) / "Weighted.mlpackage"
    model.save(str(package))
    weights = (package / "Data/com.apple.CoreML/weights/weight.bin").read_bytes()


def rust_bytes(name, data):
    lines = ["    " + ", ".join(map(str, data[i:i + 20])) + ","
             for i in range(0, len(data), 20)]
    return "pub const " + name + ": &[u8] = &[\n" + "\n".join(lines) + "\n];\n"


pathlib.Path("tests/fixtures/in_memory_program.rs").write_text(
    "//! Core ML Tools 9.0 ML Program: fp16 x multiplied by a 2x identity matrix.\n"
    "//! Generated from scripts/generate-memory-fixture.py; weights are external.\n"
    + rust_bytes("MODEL", specification) + rust_bytes("WEIGHTS", weights)
)
