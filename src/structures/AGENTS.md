# STRUCTURES MODULE

Core tensor and layer implementations. **COMPLEXITY HOTSPOT** (tensor.rs 703L, layer.rs 531L).

## OVERVIEW

Tensor operations + neural network layers with pyo3 Python bindings. Forward/backward propagation, batch processing, shape manipulation.

## FILES

- `tensor.rs` (703L): Tensor class - construction, NumPy interop, arithmetic ops, reshape/transpose/concat, serialization, iteration
- `layer.rs` (531L): Layer class - forward/backward propagation, batch processing, activation/cost selection
- `tensor_ops.rs`: Private tensor operations (add/sub/mul/div implementations)
- `tenosor_iter.rs`: ⚠️ TYPO - should be tensor_iter.rs - Iterator implementations
- `mod.rs`: Module exports

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Shape bugs** | layer.rs forward/backward | Batch transposes, mean_axis, shape assumptions |
| **Arithmetic bugs** | tensor.rs operators | __add__/__mul__/__truediv__, dot() 1D/2D branching |
| **Reshape/transpose** | tensor.rs | reshape(), transpose(), flatten(), push_row/column |
| **Serialization** | tensor.rs/layer.rs | to_dict(), from_dict() - currently panic! on errors |
| **Batch processing** | layer.rs forward_batch/backward | Transpose logic, delta calculations |
| **New layer types** | layer.rs | Add to Layer::new(), update forward/backward |

## ANTI-PATTERNS

**CRITICAL (present in this module):**
1. ❌ `panic!()` in tensor.rs, layer.rs - MUST return PyResult
2. ❌ `unwrap()`/`expect()` in runtime paths - MUST propagate errors
3. ❌ `println!()` in tensor.rs from_dict - MUST use logging
4. ❌ Panicking during deserialization - MUST return PyErr
5. ❌ dot() assumes 1D/2D only - explicit dimension check missing

**Shape assumptions:**
- layer.rs assumes batch dimension is first axis
- Many operations clone tensors unnecessarily
- Transposes are fragile (forward_batch, backward)

## NOTES

**Typo:** `tenosor_iter.rs` → should be `tensor_iter.rs`

**Key exports (via #[pymethods]):**
- Tensor: new, zeros, random, from_numpy, dot, reshape, transpose, flatten, cat, to_dict, from_dict
- Layer: new, forward, forward_batch, backward, get_output_deltas, to_dict, from_dict

**Private modules:** tensor_ops, tenosor_iter - not exposed to Python, only used internally
