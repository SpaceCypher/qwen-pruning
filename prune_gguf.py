"""Remove quantization-aligned MLP channels from a Qwen3.5-4B GGUF."""
import argparse
import json
from pathlib import Path
import re

import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--strategy", choices=("norm", "random"), required=True)
    parser.add_argument("--drop-blocks", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    reader = gguf.GGUFReader(args.source)
    arch = reader.get_field("general.architecture").contents()
    width_key = f"{arch}.feed_forward_length"
    width_field = reader.get_field(width_key)
    width = width_field.contents()
    if arch != "qwen35" or width != 9216 or not 0 < args.drop_blocks < width // 256:
        parser.error("expected Qwen3.5 with MLP width 9216 and 1–35 blocks to remove")

    block = 256
    width_new = width - block * args.drop_blocks
    rng = np.random.default_rng(42)
    keep = {}
    for tensor in reader.tensors:
        if not re.fullmatch(r"blk\.\d+\.ffn_down\.weight", tensor.name):
            continue
        layer = tensor.name.split(".")[1]
        if args.strategy == "norm":
            weights = gguf.dequantize(tensor.data, tensor.tensor_type)
            scores = np.linalg.norm(weights.reshape(weights.shape[0], -1, block), axis=(0, 2))
            indices = np.argsort(scores, kind="stable")[args.drop_blocks:]
        else:
            indices = rng.permutation(width // block)[args.drop_blocks:]
        keep[layer] = np.sort(indices)

    writer = gguf.GGUFWriter(args.output, arch, endianess=reader.endianess)
    alignment = reader.get_field("general.alignment")
    if alignment:
        writer.data_alignment = alignment.contents()
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        value = width_new if field.name == width_key else field.contents()
        if value is not None:
            writer.add_key_value(field.name, value, kind, sub_type=subtype)

    def tensor_data(tensor):
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight", tensor.name)
        if not match:
            return tensor.data
        layer, projection = match.groups()
        selected = keep[layer]
        if projection == "down":
            rows = tensor.data.reshape(tensor.data.shape[0], width // block, -1)
            return np.ascontiguousarray(rows[:, selected, :].reshape(rows.shape[0], -1))
        channels = (selected[:, None] * block + np.arange(block)).ravel()
        return np.ascontiguousarray(tensor.data[channels])

    for tensor in reader.tensors:
        data = tensor_data(tensor)
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor_data(tensor), tensor_endianess=reader.endianess)
    writer.close()
    result = gguf.GGUFReader(args.output)
    assert result.get_field(width_key).contents() == width_new
    assert len(result.tensors) == len(reader.tensors)
    assert args.output.stat().st_size < args.source.stat().st_size
    manifest = {
        "strategy": args.strategy, "original_width": width, "pruned_width": width_new,
        "block_width": block, "source_bytes": args.source.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "kept_blocks": {layer: indices.tolist() for layer, indices in keep.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({key: value for key, value in manifest.items() if key != "kept_blocks"})


if __name__ == "__main__":
    main()
