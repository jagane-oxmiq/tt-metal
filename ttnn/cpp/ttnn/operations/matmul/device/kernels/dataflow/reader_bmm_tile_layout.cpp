// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

// MXFP4 to FP32 conversion function
inline void convert_mxfp4_to_fp32_tile(uint32_t* mxfp4_tile, uint32_t* fp32_tile, uint32_t tile_size) {
    // MXFP4 format: 4-bit mantissa with shared 8-bit exponent per block
    // Conversion: Extract 4-bit values and reconstruct FP32
    constexpr uint32_t MXFP4_BLOCK_SIZE = 32;  // 32 4-bit values share one exponent

    for (uint32_t i = 0; i < tile_size / MXFP4_BLOCK_SIZE; i++) {
        // Extract shared exponent (stored at beginning of each block)
        uint8_t shared_exp = (mxfp4_tile[i * 5] >> 24) & 0xFF;  // First byte contains shared exponent

        // Process 32 4-bit values in the block
        for (uint32_t j = 0; j < MXFP4_BLOCK_SIZE; j++) {
            uint32_t byte_idx = (i * 5) + (j / 8) + 1;  // Skip exponent byte
            uint32_t nibble_idx = j % 8;

            // Extract 4-bit mantissa
            uint8_t mxfp4_val = (mxfp4_tile[byte_idx] >> (nibble_idx * 4)) & 0x0F;

            // Convert to FP32
            if (mxfp4_val == 0) {
                fp32_tile[i * MXFP4_BLOCK_SIZE + j] = 0;
            } else {
                // Reconstruct FP32: sign=0, exp=shared_exp, mantissa from 4-bit value
                uint32_t fp32_val = ((uint32_t)shared_exp << 23) | ((uint32_t)mxfp4_val << 19);
                fp32_tile[i * MXFP4_BLOCK_SIZE + j] = fp32_val;
            }
        }
    }
}

void kernel_main() {
    bool one_time_profile = true;

    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks = get_arg_val<uint32_t>(16);

    // batch args
    uint32_t MtKt = get_arg_val<uint32_t>(17);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(18);
    uint32_t batch = get_arg_val<uint32_t>(19);
    uint32_t bcast_B = get_arg_val<uint32_t>(20);

    // MXFP4 quantization flags (new args)
    uint32_t in0_is_mxfp4 = get_arg_val<uint32_t>(21);
    uint32_t in1_is_mxfp4 = get_arg_val<uint32_t>(22);

    constexpr auto in0_args = TensorAccessorArgs<0>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    const auto s0 = TensorAccessor(in0_args, in0_tensor_addr, in0_single_tile_size_bytes);
    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
        uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_reserve_back(cb_id_in0, in0_block_num_tiles);
            cb_reserve_back(cb_id_in1, in1_block_num_tiles);

            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            uint32_t in0_tensor_row_start_tile_id = in0_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in0_block_h; h++) {
                uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);

                    // Convert MXFP4 to FP32 if needed
                    if (in0_is_mxfp4) {
                        uint32_t* tile_ptr = (uint32_t*)l1_write_addr_in0;
                        uint32_t tile_elements = in0_single_tile_size_bytes / sizeof(uint32_t);
                        convert_mxfp4_to_fp32_tile(tile_ptr, tile_ptr, tile_elements);
                    }

                    l1_write_addr_in0 += in0_single_tile_size_bytes;
                    in0_tensor_tile_id += in0_tensor_stride_w;
                }
                in0_tensor_row_start_tile_id += in0_tensor_stride_h;
            }
            in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; h++) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; w++) {
                    noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);

                    // Convert MXFP4 to FP32 if needed
                    if (in1_is_mxfp4) {
                        uint32_t* tile_ptr = (uint32_t*)l1_write_addr_in1;
                        uint32_t tile_elements = in1_single_tile_size_bytes / sizeof(uint32_t);
                        convert_mxfp4_to_fp32_tile(tile_ptr, tile_ptr, tile_elements);
                    }

                    l1_write_addr_in1 += in1_single_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);
            cb_push_back(cb_id_in1, in1_block_num_tiles);
        }
        if (bcast_B == 0) {
            in1_tensor_start_tile_id += KtNt;
        }
        in0_tensor_start_tile_id += MtKt;
    }
}
