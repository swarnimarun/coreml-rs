"""Convert Apple's MobileNetV2 classifier to an array-to-array benchmark model.

Requires coremltools 9. Input is already-normalized RGB CHW floats in [-1, 1];
the output is the original 1,001-element softmax vector. This removes only the
image preprocessing and classifier dictionary wrapper, leaving network weights
and layers intact. The generated model is for benchmarking, not distribution.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import coremltools as ct
from coremltools.proto import FeatureTypes_pb2, NeuralNetwork_pb2

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
expected = "c76832208ff4c936365f0f2609f7b77f7f1a6caf62b0b429056d5ad7e48635ad"
if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
    raise SystemExit("source does not match the documented MobileNetV2FP16 download")
model = ct.utils.load_spec(str(args.source))
if model.WhichOneof("Type") != "neuralNetworkClassifier":
    raise SystemExit("expected Apple's MobileNetV2 neural-network classifier")
classifier = copy.deepcopy(model.neuralNetworkClassifier)
if len(classifier.stringClassLabels.vector) != 1001 or classifier.labelProbabilityLayerName != "classLabelProbs":
    raise SystemExit("unexpected MobileNetV2 interface")
labels = list(classifier.stringClassLabels.vector)
if any("\n" in label or "\r" in label for label in labels):
    raise SystemExit("class label contains a line break")
if len(classifier.preprocessing) != 1 or classifier.preprocessing[0].featureName != "image":
    raise SystemExit("unexpected image preprocessing")
scaler = classifier.preprocessing[0].scaler
if abs(scaler.channelScale - 2.0 / 255.0) > 1e-7 or any(abs(v + 1.0) > 1e-7 for v in (scaler.redBias, scaler.greenBias, scaler.blueBias)):
    raise SystemExit("unexpected image normalization")
network = NeuralNetwork_pb2.NeuralNetwork()
network.layers.extend(classifier.layers)
model.ClearField("neuralNetworkClassifier")
model.neuralNetwork.CopyFrom(network)
input_feature = model.description.input[0]
if input_feature.name != "image":
    raise SystemExit("unexpected input name")
input_feature.type.Clear()
input_feature.type.multiArrayType.shape.extend([3, 224, 224])
input_feature.type.multiArrayType.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT32
model.description.ClearField("output")
output = model.description.output.add()
output.name = "classLabelProbs"
output.type.multiArrayType.shape.extend([1001])
output.type.multiArrayType.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT32
model.description.ClearField("predictedFeatureName")
model.description.ClearField("predictedProbabilitiesName")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(model.SerializeToString(deterministic=True))
label_path = args.output.with_suffix(".labels.txt")
label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
print(f"wrote {args.output} ({args.output.stat().st_size} bytes) and {label_path}")
