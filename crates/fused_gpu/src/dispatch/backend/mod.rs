use core::fmt::Debug;
use std::vec::Vec;

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend, TargetCompilationOptions, TargetFlags, backend::kernel::{LinkedKernel, RawKernel},
    }, errors::{Error, ErrorKind}, tensor::{Block, bf16, f16},
};

pub mod codegen;

pub mod kernel;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "rocm")]
pub mod rocm;

mod graph;

pub use graph::*;

mod ops;

pub use ops::Op;

#[cfg(feature = "wgsl")]
pub type GpuContext = wgsl::GpuContext;
#[cfg(not(feature = "wgsl"))]
pub type GpuContext = NopGpuContext;

#[cfg(feature = "wgsl")]
pub type GpuBuffer = wgsl::Buffer;
#[cfg(not(feature = "wgsl"))]
pub type GpuBuffer = ();

impl GpuBufferBackend for () {
    fn size(&self) -> u32 {
        0
    }

    fn size_bytes(&self) -> u32 {
        0
    }
}

impl GpuKernelBackend for () {
    fn block(&self) -> Block {
        Block {
            dim3: [0; 3],
            tile: [0; 3],
        }
    }

    fn iteration_space(&self) -> &[MetaId] {
        &[]
    }
}

#[derive(Debug, Clone)]
pub struct NopGpuContext;

impl NopGpuContext {
    pub const fn new() -> Result<Self, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }
}

impl GpuBackend for NopGpuContext {
    type Buffer = ();
    type MetaBuf = ();
    type Kernel = ();
    type Schedule<'a> = ();

    fn target_spec(&self) -> TargetCompilationOptions {
        TargetCompilationOptions {
            flags: TargetFlags::empty(),
            subgroup_size: None,
        }
    }

    fn alloc(&self, _len: u32) -> Result<Self::Buffer, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn alloc_zeroed(&self, _len: u32) -> Result<Self::Buffer, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn alloc_init(&self, _data: &[u8]) -> Result<Self::Buffer, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn alloc_meta(&self, _data: &[u32]) -> Result<Self::MetaBuf, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn compile(
        &self,
        _src: &RawKernel,
        _params: &[Param],
        _options: &CompilationOptions,
    ) -> Result<Self::Kernel, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn download(
        &self,
        _buffer: &Self::Buffer,
        _out: &mut [u8],
        _src_off: u32,
    ) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn schedule_parallel<'a>(
        &self,
        _kernel: &Self::Kernel,
        _bindings: &[Vec<&'a Self::Buffer>],
        _meta: &[u32],
        _meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule<'a>, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn dispatch_kernel(
        &self,
        _kernel: &Self::Kernel,
        _wg: [u32; 3],
        _bindings: &[&Self::Buffer],
        _meta: &Self::MetaBuf,
    ) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn dispatch_schedule(&self, _schedule: &Self::Schedule<'_>) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn schedule(
        &self,
        _kernels: Vec<kernel::Dependencies<kernel::Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        _bindings: &[&Self::Buffer],
        _meta: &[u32],
        _meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule<'_>, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn is_ready(&self) -> Result<bool, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn sync(&self) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn upload(&self, _buffer: &Self::Buffer, _data: &[u8], _dst_off: u32) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn copy(&self, _src: &Self::Buffer, _dst: &Self::Buffer) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum ConstOrParam {
    Param(NodeId),

    ConstU32(u32),
    ConstI32(i32),
    ConstF32(f32),
    ConstF16(f16),
    ConstBf16(bf16),

    Neither,
}

impl LinkedKernel {
    pub const fn try_register_param(&mut self, maybe_param: ConstOrParam) {
        if let ConstOrParam::Param(id) = maybe_param {
            self.register_param(id);
        }
    }
}
