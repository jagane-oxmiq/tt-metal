// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <stdint.h>

// FP32 to MXFP4 conversion function
inline void convert_fp32_to_mxfp4_tile(uint32_t* fp32_tile, uint32_t* mxfp4_tile, uint32_t tile_size) {
    // MXFP4 format: 4-bit mantissa with shared 8-bit exponent per block
    constexpr uint32_t MXFP4_BLOCK_SIZE = 32;  // 32 4-bit values share one exponent

    for (uint32_t i = 0; i < tile_size / MXFP4_BLOCK_SIZE; i++) {
        // Find the maximum exponent in the block for shared scaling
        uint8_t max_exp = 0;
        for (uint32_t j = 0; j < MXFP4_BLOCK_SIZE; j++) {
            uint32_t fp32_val = fp32_tile[i * MXFP4_BLOCK_SIZE + j];
            if (fp32_val != 0) {
                uint8_t exp = (fp32_val >> 23) & 0xFF;
                if (exp > max_exp) {
                    max_exp = exp;
                }
            }
        }

        // Store shared exponent at the beginning of the block
        mxfp4_tile[i * 5] = ((uint32_t)max_exp << 24);

        // Convert each FP32 value to 4-bit mantissa
        for (uint32_t j = 0; j < MXFP4_BLOCK_SIZE; j++) {
            uint32_t fp32_val = fp32_tile[i * MXFP4_BLOCK_SIZE + j];
            uint8_t mxfp4_val = 0;

            if (fp32_val != 0) {
                // Extract mantissa and scale according to shared exponent
                uint32_t mantissa = (fp32_val & 0x7FFFFF) >> 19;  // Take top 4 bits of mantissa
                uint8_t exp = (fp32_val >> 23) & 0xFF;

                // Scale mantissa based on exponent difference
                int exp_diff = max_exp - exp;
                if (exp_diff >= 0 && exp_diff < 16) {
                    mxfp4_val = mantissa >> exp_diff;
                }
            }

            // Pack 4-bit value into output
            uint32_t byte_idx = (i * 5) + (j / 8) + 1;  // Skip exponent byte
            uint32_t nibble_idx = j % 8;
            mxfp4_tile[byte_idx] |= ((uint32_t)mxfp4_val << (nibble_idx * 4));
        }
    }
}

void kernel_main() {
    // out tensor args
    uint32_t out_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(4);
    uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(5);

    // out subblock args
    uint32_t out_subblock_w = get_arg_val<uint32_t>(6);
    uint32_t out_subblock_h = get_arg_val<uint32_t>(7);
    uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(8);
    uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(9);
    uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(10);

    // batch args
    uint32_t MtNt = get_arg_val<uint32_t>(11);  // if 0
    uint32_t batch = get_arg_val<uint32_t>(12);

    // MXFP4 quantization flag (new arg)
    uint32_t out_is_mxfp4 = get_arg_val<uint32_t>(13);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_out0);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, out_tensor_addr, single_tile_size_bytes);

    bool one_time_profile = true;
    for (uint32_t b = 0; b < batch; b++) {
        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

                cb_wait_front(cb_id_out0, out_subblock_tile_count);
                uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        // Convert FP32 to MXFP4 if needed before writing
                        if (out_is_mxfp4) {
                            uint32_t* tile_ptr = (uint32_t*)l1_read_addr;
                            uint32_t tile_elements = single_tile_size_bytes / sizeof(uint32_t);

                            // Create temporary buffer for MXFP4 data (smaller than FP32)
                            // MXFP4 uses ~5 bytes per 32 values (1 byte exp + 4 bytes for 32 4-bit values)
                            uint32_t mxfp4_buffer[tile_elements / 6];  // Approximate size
                            convert_fp32_to_mxfp4_tile(tile_ptr, mxfp4_buffer, tile_elements);

                            // Write the compressed MXFP4 data
                            uint32_t mxfp4_size = (tile_elements / 32) * 5 * sizeof(uint32_t);
                            noc_async_write(out_tensor_tile_id, s.address, l1_read_addr, mxfp4_size);
                        } else {
                            noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
                        }
                        l1_read_addr += single_tile_size_bytes;

                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
}
