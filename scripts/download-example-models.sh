#!/bin/sh
set -eu
mkdir -p demo/models
fetch() {
    name=$1
    expected=$2
    url=$3
    if [ ! -f "demo/models/$name" ]; then
        tmp=$(mktemp "demo/models/.download.XXXXXX")
        trap 'rm -f "$tmp"' EXIT HUP INT TERM
        curl --fail --location --retry 2 --output "$tmp" "$url"
        printf '%s  %s\n' "$expected" "$tmp" | shasum -a 256 --check --status
        mv "$tmp" "demo/models/$name"
        trap - EXIT HUP INT TERM
    fi
    printf '%s  %s\n' "$expected" "demo/models/$name" | shasum -a 256 --check
}
fetch MobileNetV2FP16.mlmodel c76832208ff4c936365f0f2609f7b77f7f1a6caf62b0b429056d5ad7e48635ad https://ml-assets.apple.com/coreml/models/Image/ImageClassification/MobileNetV2/MobileNetV2FP16.mlmodel
fetch FastViTT8F16.mlpackage.zip 156df7e15dd384af30727f1eee4982602924b7e8ae37a65cefa1cfb325261508 https://ml-assets.apple.com/coreml/models/Image/ImageClassification/FastViT/FastViTT8F16.mlpackage.zip
fetch DeepLabV3FP16.mlmodel 6c2c8cbe1c2d765f231408458b8c273976039ae04a05b112d748fc21cb3bfe95 https://ml-assets.apple.com/coreml/models/Image/ImageSegmentation/DeepLabV3/DeepLabV3FP16.mlmodel
fetch dog.jpg f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
if [ ! -d demo/models/FastViTT8F16.mlpackage ]; then
    unzip -q demo/models/FastViTT8F16.mlpackage.zip 'FastViTT8F16.mlpackage/*' -d demo/models
fi
