"""Regenerate the stateful test spec with coremltools 9 and Python 3.13.

This is a development-only tool. The resulting source fixture has no runtime
Python dependency. Run from the repo root and inspect the diff afterward.
"""
import pathlib

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


@mb.program(
    input_specs=[
        mb.TensorSpec((1,), dtype=types.fp16),
        mb.StateTensorSpec((1,), dtype=types.fp16),
    ],
    opset_version=ct.target.iOS18,
)
def accumulator_model(x, accumulator):
    previous = mb.read_state(input=accumulator)
    y = mb.add(x=x, y=previous, name="y")
    mb.coreml_update_state(state=accumulator, value=y)
    return y


model = ct.convert(
    accumulator_model,
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,
)
# Conversion timestamps are not relevant to the test and make output vary.
spec = model.get_spec()
spec.description.metadata.userDefined.pop("com.github.apple.coremltools.conversion_date", None)
data = spec.SerializeToString(deterministic=True)
lines = ["    " + ", ".join(map(str, data[i:i + 20])) + "," for i in range(0, len(data), 20)]
pathlib.Path("tests/fixtures/stateful.rs").write_text(
    "//! Core ML Tools 9.0 ML Program: fp16 accumulator, x + state -> y.\n"
    "//! Generated from scripts/generate-state-fixture.py; no weights.\n"
    "pub const MODEL: &[u8] = &[\n" + "\n".join(lines) + "\n];\n"
)
