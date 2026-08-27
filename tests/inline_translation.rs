//! Differential tests for `Config::enable_inline_address_translation`.
//!
//! The `test_interpreter_and_jit_asm!` macro runs every program through the interpreter (which
//! never uses the inline fast path and is therefore the semantic reference) and through the JIT
//! (which does), asserting identical results, instruction meters, final pc, and register traces.
//! Every test here compiles with the inline fast path enabled, so a divergence between the two
//! engines is a fast-path bug by construction.
//!
//! Coverage: all access sizes on loads and register stores, zero-extension, region bounds at the
//! exact boundary, the read-only store rejection, the gapped stack (hit, gap violation, and a
//! read spanning two host-contiguous frames), gaps-off stack, out-of-range region indices, the
//! empty heap, and the register-aliasing corners of the emission (value in RBX/R12, address ==
//! value, load dst == address source).

#![allow(clippy::literal_string_with_formatting_args)]
#![allow(clippy::arithmetic_side_effects)]
#![cfg(all(feature = "jit", not(target_os = "windows"), target_arch = "x86_64"))]

use solana_sbpf::{
    assembler::assemble,
    ebpf,
    error::{EbpfError, ProgramResult},
    memory_region::{AccessType, MemoryRegion},
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    static_analysis::Analysis,
    verifier::RequisiteVerifier,
    vm::{Config, ContextObject},
};
use std::sync::Arc;
use test_utils::{
    compare_register_trace, create_vm, test_interpreter_and_jit, test_interpreter_and_jit_asm,
    TestContextObject,
};

fn inline_config(sbpf_version: SBPFVersion) -> Config {
    Config {
        aligned_memory_mapping: true,
        enable_inline_address_translation: true,
        enabled_sbpf_versions: sbpf_version..=sbpf_version,
        ..Config::default()
    }
}

#[test]
fn test_inline_loads_all_sizes() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        // ldxb of a high-bit byte: also proves zero-extension, not sign-extension.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxb r0, [r1+0]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0x11, 0xcc, 0xdd],
            TestContextObject::new(3),
            ProgramResult::Ok(0xaa),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxh r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0x11, 0x22, 0xcc, 0xdd],
            TestContextObject::new(3),
            ProgramResult::Ok(0x2211),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0x11, 0x22, 0x33, 0x44, 0xcc, 0xdd],
            TestContextObject::new(3),
            ProgramResult::Ok(0x44332211),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxdw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xcc, 0xdd],
            TestContextObject::new(3),
            ProgramResult::Ok(0x8877665544332211),
        );
    }
}

#[test]
fn test_inline_stores_all_sizes() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0x11
            stxb [r1+2], r2
            ldxdw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0xff, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xcc, 0xdd],
            TestContextObject::new(5),
            ProgramResult::Ok(0x8877665544332211),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0x2211
            stxh [r1+2], r2
            ldxdw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0xff, 0xff, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xcc, 0xdd],
            TestContextObject::new(5),
            ProgramResult::Ok(0x8877665544332211),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0x44332211
            stxw [r1+2], r2
            ldxdw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0xff, 0xff, 0xff, 0xff, 0x55, 0x66, 0x77, 0x88, 0xcc, 0xdd],
            TestContextObject::new(5),
            ProgramResult::Ok(0x8877665544332211),
        );
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0x44332211
            stxdw [r1+2], r2
            ldxdw r0, [r1+2]
            exit",
            config.clone(),
            [0xaa, 0xbb, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xcc, 0xdd],
            TestContextObject::new(5),
            ProgramResult::Ok(0x44332211),
        );
    }
}

#[test]
fn test_inline_input_boundary() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        // Last valid byte.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxb r0, [r1+15]
            exit",
            config.clone(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x5a],
            TestContextObject::new(3),
            ProgramResult::Ok(0x5a),
        );
        // Last valid dw.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxdw r0, [r1+8]
            exit",
            config.clone(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0, 0, 0, 0, 0, 0, 0x88],
            TestContextObject::new(3),
            ProgramResult::Ok(0x8800000000000011),
        );
        // One byte past the end.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxdw r0, [r1+9]
            exit",
            config.clone(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            TestContextObject::new(2),
            ProgramResult::Err(EbpfError::AccessViolation(
                AccessType::Load,
                0x400000009,
                8,
                "input"
            )),
        );
        // Store one byte past the end. The address begins outside the region, which the
        // stock naming reports as "unallocated".
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 1
            stxb [r1+16], r2
            exit",
            config.clone(),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            TestContextObject::new(3),
            ProgramResult::Err(EbpfError::AccessViolation(
                AccessType::Store,
                0x400000010,
                1,
                "unallocated"
            )),
        );
    }
}

#[test]
fn test_inline_store_to_rodata() {
    // V0 maps its rodata at MM_BYTECODE_START; a register store there must take the slow path
    // (store table row is zeroed for read-only regions) and produce the stock violation.
    let config = inline_config(SBPFVersion::V0);
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r2, 7
        mov64 r1, 1
        lsh64 r1, 32
        stxb [r1], r2
        exit",
        config.clone(),
        [],
        TestContextObject::new(5),
        ProgramResult::Err(EbpfError::AccessViolation(
            AccessType::Store,
            0x100000000,
            1,
            "program"
        )),
    );
}

#[test]
fn test_inline_stack_roundtrip() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0x11223344
            stxdw [r10-8], r2
            ldxdw r0, [r10-8]
            exit",
            config.clone(),
            [],
            TestContextObject::new(5),
            ProgramResult::Ok(0x11223344),
        );
    }
}

#[test]
fn test_inline_stack_gap_violation_v0() {
    // V0 with frame gaps on: MM_STACK_START + frame_size + 8 lands in the first gap.
    let config = inline_config(SBPFVersion::V0);
    let addr = ebpf::MM_STACK_START + config.stack_frame_size as u64 + 8;
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r1, 2
        lsh64 r1, 32
        add64 r1, 4104
        ldxdw r0, [r1]
        exit",
        config.clone(),
        [],
        TestContextObject::new(5),
        ProgramResult::Err(EbpfError::StackAccessViolation(AccessType::Load, addr, 8, 1)),
    );
}

#[test]
fn test_inline_stack_no_gaps_v4() {
    // V4 builds a non-gapped stack, so the same address is plain zero-filled stack memory and
    // the access goes through the generic descriptor path.
    let config = inline_config(SBPFVersion::V4);
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r1, 2
        lsh64 r1, 32
        add64 r1, 4104
        ldxdw r0, [r1]
        exit",
        config.clone(),
        [],
        TestContextObject::new(6),
        ProgramResult::Ok(0),
    );
}

#[test]
fn test_inline_stack_gaps_disabled_config() {
    // Same layout question from the config side: gaps off makes the V0 stack non-gapped, the
    // specialized stack block is not emitted, and the generic path serves region 2.
    let config = Config {
        enable_stack_frame_gaps: false,
        ..inline_config(SBPFVersion::V0)
    };
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r1, 2
        lsh64 r1, 32
        add64 r1, 4104
        ldxdw r0, [r1]
        exit",
        config.clone(),
        [],
        TestContextObject::new(6),
        ProgramResult::Ok(0),
    );
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r2, 0x11223344
        stxdw [r10-8], r2
        ldxdw r0, [r10-8]
        exit",
        config.clone(),
        [],
        TestContextObject::new(5),
        ProgramResult::Ok(0x11223344),
    );
}

#[test]
fn test_inline_stack_cross_frame_read_v0() {
    // A dw read beginning in the last 4 bytes of frame 0 is admitted by the stock gap
    // arithmetic and reads into frame 1's host bytes (frames are host-contiguous). The inline
    // path must reproduce it bit for bit.
    let config = inline_config(SBPFVersion::V0);
    test_interpreter_and_jit_asm!(
        "
        add64 r10, 0
        mov64 r2, 0x7abbccdd
        stxw [r10-4], r2
        mov64 r3, 2
        lsh64 r3, 32
        add64 r3, 8192
        mov64 r4, 0x11223344
        stxw [r3], r4
        ldxdw r0, [r10-4]
        exit",
        config.clone(),
        [],
        TestContextObject::new(10),
        ProgramResult::Ok(0x112233447abbccdd),
    );
}

#[test]
fn test_inline_region_out_of_range() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r1, 9
            lsh64 r1, 32
            ldxdw r0, [r1]
            exit",
            config.clone(),
            [],
            TestContextObject::new(4),
            ProgramResult::Err(EbpfError::AccessViolation(
                AccessType::Load,
                0x900000000,
                8,
                "unallocated"
            )),
        );
    }
}

#[test]
fn test_inline_empty_heap() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r1, 3
            lsh64 r1, 32
            ldxdw r0, [r1]
            exit",
            config.clone(),
            [],
            TestContextObject::new(4),
            ProgramResult::Err(EbpfError::AccessViolation(
                AccessType::Load,
                0x300000000,
                8,
                "unallocated"
            )),
        );
    }
}

#[test]
fn test_inline_store_value_register_aliasing() {
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        // Value in guest r6 (host RBX): the store path must borrow a different temporary.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r6, 0x66
            stxb [r1+0], r6
            ldxb r0, [r1+0]
            exit",
            config.clone(),
            [0xff],
            TestContextObject::new(5),
            ProgramResult::Ok(0x66),
        );
        // Value in guest r7 (host R12): the alternative temporary.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r7, 0x77
            stxb [r1+0], r7
            ldxb r0, [r1+0]
            exit",
            config.clone(),
            [0xff],
            TestContextObject::new(5),
            ProgramResult::Ok(0x77),
        );
        // Address and value in the same register.
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r6, 4
            lsh64 r6, 32
            stxdw [r6], r6
            ldxdw r0, [r6]
            exit",
            config.clone(),
            [0, 0, 0, 0, 0, 0, 0, 0],
            TestContextObject::new(6),
            ProgramResult::Ok(0x400000000),
        );
    }
}

#[test]
fn test_inline_byte_store_from_high_byte_alias_registers() {
    // Guest r1 maps to host RSI: an 8-bit store from encodings 4-7 needs a REX prefix or the
    // JIT silently stores AH/CH/DH/BH instead (the x86.rs force_rex fix). r2 (RDX, encoding 2)
    // is loaded with a poison value whose bits 8-15 would leak if DH were stored.
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0xEE00
            mov64 r3, r1
            mov64 r1, 0x5c
            stxb [r3+0], r1
            ldxb r0, [r3+0]
            exit",
            config.clone(),
            [0xff],
            TestContextObject::new(7),
            ProgramResult::Ok(0x5c),
        );
        // The same store through the slow-path anchor (stack region byte store from r1).
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            mov64 r2, 0xEE00
            mov64 r1, 0x7d
            stxb [r10-1], r1
            ldxb r0, [r10-1]
            exit",
            config.clone(),
            [],
            TestContextObject::new(6),
            ProgramResult::Ok(0x7d),
        );
    }
}

#[test]
fn test_inline_load_dst_is_address_source() {
    // dst doubles as the fast path's temporary; dst == the address source register is the
    // aliasing corner (the guest address lives in REGISTER_SCRATCH by then).
    for sbpf_version in [SBPFVersion::V0, SBPFVersion::V4] {
        let config = inline_config(sbpf_version);
        test_interpreter_and_jit_asm!(
            "
            add64 r10, 0
            ldxdw r1, [r1+0]
            mov64 r0, r1
            exit",
            config.clone(),
            [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
            TestContextObject::new(4),
            ProgramResult::Ok(0x8877665544332211),
        );
    }
}
