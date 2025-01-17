# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31
# modifyed by leifang on 2023.08.11

import os
import argparse
from typing import Dict, List, Tuple
from .enum_defines import Platform, MemType
from .combine import graph_adapter
from .graph_serialize import serialize
from .load_model import load_onnx_model
from .graph_optimizer import graph_optimizer
from .generate_report import build_memory_plan_report, remove_invalid_file

platform_list = [s.name for s in Platform]
memory_list = [s.name for s in MemType]

def str2bool(str:str):
    return True if str.lower() == 'true' else False

def main():
    parser = argparse.ArgumentParser(
        description="welcome to use thinker offline tools", add_help=True
    )

    parser.add_argument(
        "-g", "--graph", required=True, type=str, help="Path of onnx graph."
    )
    parser.add_argument(
        "-p",
        "--platform",
        default="venus",
        type=str,
        help="target platform",
        choices=platform_list,
    )
    parser.add_argument(
        "-m",
        "--memory",
        default=None,
        type=str,
        help="set params and entries location. Example:  params:psram, inputs[0]:share-memory",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        default=None,
        type=str,
        help="ignore methods of op fusion",
        choices=["Remove_QuantDequant"],
    )
    parser.add_argument(
        "-d", 
        "--dump", 
        default='False', 
        type=str2bool,
        help="dump middle onnx or not.",
        choices=[True, False]
    )

    parser.add_argument(
        "-o", 
        "--output", 
        default="model.pkg", 
        type=str, 
        help="Path of output file."
    )

    args = parser.parse_args()

    remove_invalid_file()

    print("=" * 83)
    print("****** load model:{} ******".format(args.graph))
    graph = load_onnx_model(args.graph, args.dump)

    print("=" * 83)
    print("****** graph optimizer ******")
    graph = graph_optimizer(graph, args.strategy, args.dump)

    print("=" * 83)
    print("****** graph adjust device ******")
    model, memory_plan = graph_adapter(graph, args.platform, args.memory, args.dump)

    print("=" * 83)
    print("****** generate memory plan report ******")
    if args.dump:
        build_memory_plan_report(args.graph, memory_plan)

    print("=" * 83)
    print("****** graph serialize ******")
    packed_model = serialize(model, memory_plan)

    with open(args.output, "wb") as f:
        f.write(packed_model.to_bytes())

    print("=" * 83)
    print("****** pack success ******")
    print("=" * 83)


if __name__ == "__main__":
    main()

__all__ = ["build_memory_plan_report"]
