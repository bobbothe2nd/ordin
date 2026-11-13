use fused_gpu::{
    dispatch::{
        CompilationOptions, backend::{
            Axis, ConstOrParam, DType, Graph, GraphOp, NodeId, Op, Param, ParamId, SimpleDType, TensorLayout, ValueId, ValueState, kernel::{LinkedKernel, NodeInput},
        },
    }, errors::{Error, ErrorKind},
};
use std::vec::Vec;

pub fn forward_matmul(
    _eval_node: impl Fn(
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        &[Option<ParamId>],
        &[Option<ParamId>],
        &mut LinkedKernel,
        ValueId,
        ValueId,
        ValueId,
        ValueId,
        u32,
        ValueId,
        &[Param],
        &mut bool,
        &CompilationOptions,
    ) -> Result<Vec<NodeId>, Error>,
    dtype: SimpleDType,
    _root: NodeId,
    _resolved: &mut Vec<NodeId>,
    backwardness: Option<u8>,
    _m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    node_id: NodeId,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &Graph,
    _out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel,
    base: ValueId,
    _row: ValueId,
    _col: ValueId,
    _local_row: ValueId,
    _local_col: ValueId,
    _shared_size: u32,
    tile_size: ValueId,
    _params: &[Param],
    _stable_iteration_space: &mut bool,
    options: &CompilationOptions,
) -> Result<Vec<NodeId>, Error> {
    let subgroup_size = options
        .target
        .subgroup_size
        .ok_or(Error {
            msg: "attempted WMMA matmul with unknown subgroup size",
            kind: ErrorKind::InternalError,
            ctx: (),
        })?
        .get() as u32;
    let block = [subgroup_size / 8, 8, 1];

    let tile_width = options.opt.tile_size;

    kernel.raw.block.tile = [tile_width, tile_width, 1];
    kernel.raw.block.dim3 = block;

    let bid_x = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::BlockId { axis: Axis::X }),
    );
    let tile_m = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::Mul {
            a: tile_size,
            b: bid_x,
        }),
    );

    let bid_y = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::BlockId { axis: Axis::Y }),
    );
    let tile_n = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Immut,
        Some(Op::Mul {
            a: tile_size,
            b: bid_y,
        }),
    );

    let a_err = Error {
        msg: "expected a_node mapping to param",
        kind: ErrorKind::UnresolvedInput,
        ctx: (),
    };
    let a = match a_node {
        NodeInput::Param { param, .. } => ConstOrParam::Param(*param),
        NodeInput::Node(node_id) => match graph.nodes[*node_id].op {
            GraphOp::Custom { .. } => ConstOrParam::Param(node_params[*node_id].or(saved_params[*node_id]).ok_or(a_err)?),
            GraphOp::Input if let Some(param) = node_params[*node_id] => ConstOrParam::Param(param),
            GraphOp::ConstI32(val) => ConstOrParam::ConstI32(val),
            GraphOp::ConstU32(val) => ConstOrParam::ConstU32(val),
            GraphOp::ConstF32(val) => ConstOrParam::ConstF32(val),
            GraphOp::ConstF16(val) => ConstOrParam::ConstF16(val),
            GraphOp::ConstBf16(val) => ConstOrParam::ConstBf16(val),
            _ => return Err(a_err),
        },
    };
    kernel.try_register_param(a);

    let b_err = Error {
        msg: "expected b_node mapping to param",
        kind: ErrorKind::UnresolvedInput,
        ctx: (),
    };
    let b = match b_node {
        NodeInput::Param { param, .. } => ConstOrParam::Param(*param),
        NodeInput::Node(node_id) => match graph.nodes[*node_id].op {
            GraphOp::Custom { .. } => ConstOrParam::Param(node_params[*node_id].or(saved_params[*node_id]).ok_or(b_err)?),
            GraphOp::Input if let Some(param) = node_params[*node_id] => ConstOrParam::Param(param),
            GraphOp::ConstI32(val) => ConstOrParam::ConstI32(val),
            GraphOp::ConstU32(val) => ConstOrParam::ConstU32(val),
            GraphOp::ConstF32(val) => ConstOrParam::ConstF32(val),
            GraphOp::ConstF16(val) => ConstOrParam::ConstF16(val),
            GraphOp::ConstBf16(val) => ConstOrParam::ConstBf16(val),
            _ => return Err(b_err),
        },
    };
    kernel.try_register_param(b);

    let c = if backwardness.is_none() {
        saved_params[node_id]
    } else {
        node_params[node_id]
    }
    .ok_or(Error {
        msg: "expected node_id mapping to param",
        kind: ErrorKind::UnresolvedOutput,
        ctx: (),
    })?;
    kernel.register_param(c);

    let deepest = Vec::new();

    let frag_a = kernel.raw.def_var(
        DType::MmaA {
            dtype,
            m: tile_width,
            n: tile_width,
            k: tile_width,
            layout: if swap_a {
                TensorLayout::ColMajor
            } else {
                TensorLayout::RowMajor
            },
        },
        ValueState::Mut,
        None,
    );
    let frag_b = kernel.raw.def_var(
        DType::MmaB {
            dtype,
            m: tile_width,
            n: tile_width,
            k: tile_width,
            layout: if swap_b {
                TensorLayout::ColMajor
            } else {
                TensorLayout::RowMajor
            },
        },
        ValueState::Mut,
        None,
    );
    let frag_c = kernel.raw.def_var(
        DType::MmaAccum {
            dtype,
            m: tile_width,
            n: tile_width,
            k: tile_width,
        },
        ValueState::Mut,
        None,
    );

    let zero = kernel.raw.def_var(
        DType::Simple(dtype),
        ValueState::Inline,
        Some(Op::ConstF32 { val: 0.0 }),
    );
    kernel.raw.fill_frag(frag_c, zero);

    let tk = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Mut,
        Some(Op::ConstU32 { val: 0 }),
    );

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        match a {
            ConstOrParam::Param(param) => {
                let a_col = if swap_a {
                    let a_row = kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Mul { a: k, b: tk }),
                    );
                    kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Add {
                            a: a_row,
                            b: tile_m,
                        }),
                    )
                } else {
                    let a_row = kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Mul { a: k, b: tile_m }),
                    );
                    kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Add { a: a_row, b: tk }),
                    )
                };
                let a_idx = kernel.raw.def_var(
                    DType::Simple(SimpleDType::U32),
                    ValueState::Inline,
                    Some(Op::Add { a: a_col, b: base }),
                );

                kernel.raw.load_matrix_sync(param, frag_a, a_idx, k);
            }
            ConstOrParam::ConstBf16(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::BF16), ValueState::Inline, Some(Op::ConstBf16 { val }));
                kernel.raw.fill_frag(frag_a, val);
            }
            ConstOrParam::ConstF16(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::F16), ValueState::Inline, Some(Op::ConstF16 { val }));
                kernel.raw.fill_frag(frag_a, val);
            }
            ConstOrParam::ConstF32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::F32), ValueState::Inline, Some(Op::ConstF32 { val }));
                kernel.raw.fill_frag(frag_a, val);
            }
            ConstOrParam::ConstI32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::I32), ValueState::Inline, Some(Op::ConstI32 { val }));
                kernel.raw.fill_frag(frag_a, val);
            }
            ConstOrParam::ConstU32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::ConstU32 { val }));
                kernel.raw.fill_frag(frag_a, val);
            }
            _ => unreachable!("expected a_node param or const input"),
        }

        match b {
            ConstOrParam::Param(param) => {
                let b_col = if swap_b {
                    let b_row = kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Mul { a: n, b: tile_n }),
                    );
                    kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Add { a: b_row, b: tk }),
                    )
                } else {
                    let b_row = kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Mul { a: n, b: tk }),
                    );
                    kernel.raw.def_var(
                        DType::Simple(SimpleDType::U32),
                        ValueState::Inline,
                        Some(Op::Add {
                            a: b_row,
                            b: tile_n,
                        }),
                    )
                };
                let b_idx = kernel.raw.def_var(
                    DType::Simple(SimpleDType::U32),
                    ValueState::Inline,
                    Some(Op::Add { a: b_col, b: base }),
                );

                kernel.raw.load_matrix_sync(param, frag_b, b_idx, n);
            }
            ConstOrParam::ConstBf16(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::BF16), ValueState::Inline, Some(Op::ConstBf16 { val }));
                kernel.raw.fill_frag(frag_b, val);
            }
            ConstOrParam::ConstF16(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::F16), ValueState::Inline, Some(Op::ConstF16 { val }));
                kernel.raw.fill_frag(frag_b, val);
            }
            ConstOrParam::ConstF32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::F32), ValueState::Inline, Some(Op::ConstF32 { val }));
                kernel.raw.fill_frag(frag_b, val);
            }
            ConstOrParam::ConstI32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::I32), ValueState::Inline, Some(Op::ConstI32 { val }));
                kernel.raw.fill_frag(frag_b, val);
            }
            ConstOrParam::ConstU32(val) => {
                let val = kernel.raw.def_var(DType::Simple(SimpleDType::U32), ValueState::Inline, Some(Op::ConstU32 { val }));
                kernel.raw.fill_frag(frag_b, val);
            }
            _ => unreachable!("expected a_node param or const input"),
        }

        kernel.raw.mma_sync(frag_c, frag_a, frag_b, frag_c);

        Ok(())
    })?;

    let c_row = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::Mul { a: tile_m, b: n }),
    );
    let c_col = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::Add {
            a: c_row,
            b: tile_n,
        }),
    );
    let c_idx = kernel.raw.def_var(
        DType::Simple(SimpleDType::U32),
        ValueState::Inline,
        Some(Op::Add { a: c_col, b: base }),
    );

    kernel
        .raw
        .store_matrix_sync(c, frag_c, c_idx, n, TensorLayout::RowMajor);

    Ok(deepest)
}
