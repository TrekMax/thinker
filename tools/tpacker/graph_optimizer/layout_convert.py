from typing import List
from ..enum_defines import Layout
from ..save_model import save_to_onnx_model
from ..graph import Graph, GraphNode, GraphEntry
from ..graph_analysis.ops.base import LayoutPerfData, create_operator

VARIABLE_TAG = "@"

def _make_layout_name(name: str, layout: Layout) -> str:
    """Generate layout-specific name for entries"""
    name = name.split(VARIABLE_TAG)[0]
    if layout == Layout.NCHW:
        return name
    return f"{name}{VARIABLE_TAG}{layout.name}"

def _init_graph(src_graph: Graph) -> List[Graph]:
    """Initialize a new graph with attributes and entries from source graph"""
    graph = Graph()
    graph.copy_attrs(src_graph)
    
    # Add input entries
    for entry in src_graph.inputs:
        new_entry = entry.clone()
        graph.add_entry(new_entry)
        graph.inputs.append(new_entry)
    
    # Add constant entries
    for entry in src_graph.entries.values():
        if entry.is_constant():
            new_entry = entry.clone()
            graph.add_entry(new_entry)
    
    return [graph]

# 强制使用的阈值：当单节点 performance <= 该值时，强制选择该选项
FORCE_LAYOUT_THRESHOLD = 5

def _beam_search(graph_list: List[Graph], beam_size: int, force_layout_perf: int = None) -> List[Graph]:
    """Select top 'beam_size' graphs with best performance

    Args:
        graph_list: List of graphs to select from
        beam_size: Number of graphs to keep
        force_layout_perf: If provided, force selection of graphs containing this performance value
    """
    if len(graph_list) <= beam_size:
        return graph_list

    # 如果指定了强制使用的 performance 值，优先选择包含该性能的 graph
    if force_layout_perf is not None:
        # 找到所有包含该性能的 graph（通过比较 performance 差值）
        forced_graphs = []
        for g in graph_list:
            # 检查是否有某个性能值匹配（允许一定的误差范围）
            if g.performance % 10 == force_layout_perf or abs(g.performance % 10 - force_layout_perf) < 2:
                forced_graphs.append(g)

        if forced_graphs:
            # 返回满足条件的 graph，按 performance 排序
            forced_graphs.sort(key=lambda g: g.performance)
            return forced_graphs[:beam_size]

    import heapq
    perfs = [graph.performance for graph in graph_list]
    indices = heapq.nsmallest(beam_size, range(len(perfs)), key=perfs.__getitem__)
    return [graph_list[i] for i in indices]

def _add_pack_node(graph: Graph, src_name: str, src_layout: Layout, dst_name: str, dst_layout: Layout) -> None:
    """Add packing node to convert tensor layout"""
    src_entry = graph.entries[src_name]
    dst_entry = src_entry.clone()
    dst_entry.name = dst_name
    dst_entry.set_graph_normal()
    dst_entry.layout = dst_layout
    graph.add_entry(dst_entry)
    # Update performance based on layout conversion
    pack_node = GraphNode("Packing", dst_name + "_packing")
    pack_node.inputs = [src_entry]
    pack_node.outputs = [dst_entry]
    pack_node.op = create_operator("Packing", {}, [src_entry.tensor], [dst_entry.tensor])
    permute_perfs = pack_node.op.get_layout_perf(graph.dynamic_shape)
    for permute_perf in permute_perfs:
        if permute_perf.inputs_layout[0] == src_layout:
            # Increase transpose cost to discourage excessive layout conversions
            graph.performance += permute_perf.performance * 2
            break
    graph.add_node(pack_node)

def _get_entry_by_name(graph: Graph, src_name: str) -> GraphEntry:
    """Find entry in graph by name"""
    name = src_name.split(VARIABLE_TAG)[0]
    for e in graph.entries.values():
        e_name = e.name.split(VARIABLE_TAG)[0]
        if e_name == name and e.name != src_name:
            return e
    raise ValueError(f"Entry {src_name} not found")

def _add_node_to_graph(graph: Graph, node: GraphNode, perf: LayoutPerfData) -> None:
    """Add node to graph with specified layout and performance"""
    new_node = node.clone()
    
    # Update input entries
    for i, entry in enumerate(new_node.inputs):
        if not entry.is_constant():
            entry.layout = perf.inputs_layout[i]
            entry.name = _make_layout_name(entry.name, entry.layout)
            
            if entry.name not in graph.entries:
                # Add packing node if layout not exists
                temp_entry = _get_entry_by_name(graph, entry.name)
                _add_pack_node(graph, temp_entry.name, temp_entry.layout, entry.name, entry.layout)
            new_node.inputs[i] = graph.entries[entry.name]
    
    # Update output entries
    for i, entry in enumerate(new_node.outputs):
        entry.layout = perf.outputs_layout[i]
        entry.name = _make_layout_name(entry.name, entry.layout)
        graph.add_entry(entry)
        new_node.outputs[i] = graph.entries[entry.name]
        
        # Ensure output is NCHW layout
        if entry.is_graph_output() and entry.layout != Layout.NCHW:
            output_layout = Layout.NCHW
            output_name = _make_layout_name(entry.name, output_layout)
            _add_pack_node(graph, entry.name, entry.layout, output_name, output_layout)
            graph.entries[output_name].set_graph_output()
    
    # Add node and update performance
    graph.add_node(new_node)
    graph.performance += perf.performance

def _packing2transpose(node: GraphNode) -> None:
    """Convert Packing node to Transpose node based on layout"""
    in_layout = node.inputs[0].layout
    out_layout = node.outputs[0].layout
    
    if in_layout == Layout.NCHW and out_layout == Layout.NHWC:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 2, 3, 1]
    elif in_layout == Layout.NHWC and out_layout == Layout.NCHW:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 3, 1, 2]
    elif (in_layout == Layout.NCHW and out_layout == Layout.NCWH) or \
         (in_layout == Layout.NCWH and out_layout == Layout.NCHW):
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 1, 3, 2]
    
    node.op = create_operator(node.op_type, node.attrs,
                             [node.inputs[0].tensor], [node.outputs[0].tensor])


def _optimize_transposes(graph: Graph) -> None:
    """
    Optimize transpose nodes: keep boundary transposes for each NCWH region.
    Strategy:
    - Find all NCHW->NCWH transposes (entrances)
    - Find all NCWH->NCHW transposes (exits)
    - For simple alternating pattern, keep only first entrance and last exit
    - Remove all intermediate transposes
    """
    default_layout = Layout.NCHW
    target_layout = Layout.NCWH

    # Collect all transpose nodes
    trans_nodes = []
    node_order = list(graph.nodes.keys())

    for node in graph.nodes.values():
        if node.op_type != "Transpose" or not node.inputs or not node.outputs:
            continue
        try:
            inp_layout = node.inputs[0].layout
            out_layout = node.outputs[0].layout
            idx = node_order.index(node.name)
            trans_nodes.append((idx, node.name, node, inp_layout, out_layout))
        except Exception as e:
            print(f"Warning: could not process transpose {node.name}: {e}")

    if len(trans_nodes) <= 4:
        return  # Already optimized or too few transposes

    # Split into entrances and exits
    nchw_to_ncwh = [(idx, name, node) for idx, name, node, inp, out in trans_nodes
                    if inp == default_layout and out == target_layout]
    ncwh_to_nchw = [(idx, name, node) for idx, name, node, inp, out in trans_nodes
                    if inp == target_layout and out == default_layout]

    if not nchw_to_ncwh or not ncwh_to_nchw:
        return

    # Sort by position
    nchw_to_ncwh.sort(key=lambda x: x[0])
    ncwh_to_nchw.sort(key=lambda x: x[0])

    # Check if graph has multiple consumers (branches) - be conservative
    multi_consumer_count = sum(1 for e in graph.entries.values() if len(e.dst_nodes) > 1)
    if multi_consumer_count > 2:
        print(f"Graph has {multi_consumer_count} multi-consumer entries, keeping all transposes")
        return

    # Check if it's a simple alternating pattern
    is_alternating = True
    if len(nchw_to_ncwh) != len(ncwh_to_nchw):
        is_alternating = False
    else:
        for i in range(len(nchw_to_ncwh)):
            if nchw_to_ncwh[i][0] > ncwh_to_nchw[i][0]:
                is_alternating = False
                break
            if i < len(ncwh_to_nchw) - 1 and ncwh_to_nchw[i][0] > nchw_to_ncwh[i+1][0]:
                is_alternating = False
                break

    if not is_alternating or len(nchw_to_ncwh) > 14:
        print(f"Complex structure detected ({len(nchw_to_ncwh)} entrances), keeping all transposes")
        return

    # Keep only first entrance and last exit
    nodes_to_keep = set()
    if nchw_to_ncwh:
        nodes_to_keep.add(nchw_to_ncwh[0][1])
    if ncwh_to_nchw:
        nodes_to_keep.add(ncwh_to_nchw[-1][1])

    # Transposes to remove
    nodes_to_remove = set()
    for idx, name, node in nchw_to_ncwh:
        if name not in nodes_to_keep:
            nodes_to_remove.add(name)
    for idx, name, node in ncwh_to_nchw:
        if name not in nodes_to_keep:
            nodes_to_remove.add(name)

    print(f"Transpose optimization: removing {len(nodes_to_remove)} transposes, keeping {len(nodes_to_keep)}")

    if not nodes_to_remove:
        return

    # Reconnect nodes to bypass removed transposes
    for node_name in nodes_to_remove:
        node = graph.nodes.get(node_name)
        if not node or not node.inputs or not node.outputs:
            continue

        src_entry = node.inputs[0]
        for out_entry in node.outputs:
            consumers = list(out_entry.dst_nodes)
            for consumer in consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp.name == out_entry.name:
                        consumer.inputs[i] = src_entry

    # Remove the transpose nodes
    for node_name in nodes_to_remove:
        if node_name in graph.nodes:
            del graph.nodes[node_name]

    graph.update()


def _post_process_graph(graph: Graph, enable_global_layout_convert: bool = True) -> None:
    """Post-process graph after layout conversion"""
    for node in graph.nodes.values():
        if node.op_type == "Packing":
            _packing2transpose(node)
        node.op.layout_convert(node.op_type)

        if node.op_type in {"Conv2dInt", "Conv1dInt", "ConvTranspose2dInt", "MaxPool", "AvgPool2dInt"}:
            for key in node.attrs.keys():
                node.attrs[key] = node.op.attrs.get(key)

    # Optimize transpose nodes: keep boundary transposes, remove intermediate ones
    if enable_global_layout_convert:
        _optimize_transposes(graph)

def _layout_convert(src_graph: Graph, enable_global_layout_convert: bool = True) -> Graph:
    """Main layout conversion function using beam search"""
    # Increase beam size to explore more layout combinations
    # Each node now returns 2 layout options (NCHW, NCWH)
    beam_size = 4
    beam_search_graphs = _init_graph(src_graph)

    for node in src_graph.nodes.values():
        if not node.op:
            raise ValueError("Node operation must not be None")
        perf_list = node.op.get_layout_perf(src_graph.dynamic_args_max)

        # 检查是否有满足约束条件的 performance (如 performance <= FORCE_LAYOUT_THRESHOLD)
        # 如果有，则只扩展满足条件的性能选项，忽略其他选项
        force_perf_list = [perf for perf in perf_list if perf.performance <= FORCE_LAYOUT_THRESHOLD]
        if force_perf_list:
            # 强制使用：只扩展满足条件的性能选项
            active_perf_list = force_perf_list
        else:
            active_perf_list = perf_list

        search_graphs_expand = []
        for perf in active_perf_list:
            for graph in beam_search_graphs:
                new_graph = graph.clone()
                _add_node_to_graph(new_graph, node, perf)
                search_graphs_expand.append(new_graph)

        beam_search_graphs = _beam_search(search_graphs_expand, beam_size)

    # Update outputs
    for graph in beam_search_graphs:
        for output in src_graph.outputs:
            graph.entries[output.name].set_graph_output()
            graph.outputs.append(graph.entries[output.name])

    best_graph = _beam_search(beam_search_graphs, 1)[0]
    best_graph.update()
    _post_process_graph(best_graph, enable_global_layout_convert)
    return best_graph

def layout_optimizer(graph: Graph, is_dump: bool = True, enable_global_layout_convert: bool = True) -> Graph:
    """
    Optimize graph layout for better performance

    Args:
        graph: Input graph to be optimized
        is_dump: Whether to save the optimized graph
        enable_global_layout_convert: Whether to enable global transpose optimization
                                      after single operator layout is determined

    Returns:
        Optimized graph with best layout configuration
    """
    new_graph = _layout_convert(graph, enable_global_layout_convert)
    if is_dump:
        save_to_onnx_model(new_graph, f"./workspace/{graph.name}/model.ignore/6_graph_layout_convert.onnx")
    return new_graph

__all__ = ["layout_optimizer"]