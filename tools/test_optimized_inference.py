"""Synthetic regression checks for the integrated inference policy; no GGUF.

Set FLASH_MOE_TEST_NATIVE to a compiled kernel path to include native checks.
"""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATH = Path(os.environ.get("FLASH_MOE_TEST_NATIVE",
                                  ROOT / "src/flash_moe/moe_quant_dot.dll"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from flash_moe import pack_expert, write_manifest, ExpertManifest, synthetic_router
from flash_moe.backend_ov import OpenVINOExpertBackend
from flash_moe.gated_deltanet import GatedDeltaNetLayer
from flash_moe.key import ExpertKey
from flash_moe.model import Qwen3MoEModel
from flash_moe.moe_layer import run_moe_sequence, run_moe_layer
from flash_moe.native_expert import NativeExpert, load_native_expert
from flash_moe.store import AsyncExpertStore


class OptimizedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(17)

    def store(self, count=16):
        records = []
        for expert in range(count):
            tensors = {name: (self.rng.standard_normal((16, 16)) * .1)
                       .astype(np.float16).tobytes()
                       for name in ("gate", "up", "down")}
            records.append(pack_expert(self.root, ExpertKey(0, expert), tensors))
        manifest = ExpertManifest.load(write_manifest(self.root, records))
        store = AsyncExpertStore(self.root, capacity=8, manifest=manifest)
        self.addCleanup(store.close)
        return store

    def backend(self, store):
        backend = OpenVINOExpertBackend(store, device="CPU", cache_capacity=1,
                                        native_decode="openvino")
        # Isolate arithmetic from the CPU plugin's default BF16 precision.
        backend.core.set_property("CPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32})
        self.addCleanup(backend.close)
        return backend

    def test_prefetch_is_bounded_and_read_failure_propagates(self):
        store = self.store()
        keys = [ExpertKey(0, index) for index in range(16)]
        self.assertEqual(store.prefetch_many(keys), 8)
        self.assertLessEqual(len(store.pending_loads), 8)
        for key in keys:
            self.assertEqual(set(store.read_all(key)), {"gate", "up", "down"})
        self.assertLessEqual(len(store.resident_keys()), 8)
        store.clear()
        with patch.object(store, "_read_expert", side_effect=OSError("failed range")):
            store.prefetch(keys[0])
            with self.assertRaisesRegex(OSError, "failed range"):
                store.read_all(keys[0])

    def test_concurrent_reads_preserve_payload(self):
        store = self.store()
        keys = [ExpertKey(0, index % 16) for index in range(128)]
        expected = {key: store._read_expert(key) for key in keys}
        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(store.read_all, keys))
        for key, value in zip(keys, values):
            self.assertEqual(value, expected[key])

    def test_eviction_keeps_live_buffers_and_outputs_owned(self):
        store = self.store()
        backend = self.backend(store)
        # Exercise the recyclable-buffer path with a decoder writing in-place.
        backend.tensor_types = {0: dict(gate=1, up=1, down=1)}
        def decode(raw, kind, size, out):
            out.reshape(-1)[:] = np.frombuffer(raw, np.float16)
            return out
        backend._dequant_fn = decode
        keys = [ExpertKey(0, index % 16) for index in range(64)]
        x = self.rng.standard_normal(16).astype(np.float32)
        expected = {key: backend.forward(key, x, 16, 16) for key in keys}
        with ThreadPoolExecutor(4) as pool:
            actual = list(pool.map(lambda key: backend.forward(key, x, 16, 16), keys))
        for key, got in zip(keys, actual):
            np.testing.assert_array_equal(got, expected[key])
        first = actual[0].copy()
        backend.forward(keys[-1], x * 2, 16, 16)
        np.testing.assert_array_equal(first, actual[0])

    def test_grouped_prefill_matches_sequential_routing(self):
        store = self.store()
        backend = self.backend(store)
        router = synthetic_router(16, 16, 4, seed=5)
        x = (self.rng.standard_normal((20, 16)) * .1).astype(np.float32)
        expected = np.stack([run_moe_layer(0, row, router, backend, 16, 16) for row in x])
        actual = run_moe_sequence(0, x, router, backend, 16, 16)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=.005)
        self.assertLessEqual(len(store.pending_loads), 8)
        compiled = backend._batch_compiled_models[(16, 16)]
        self.assertGreater(int(compiled.get_property("INFERENCE_NUM_THREADS")), 0)
        self.assertEqual(backend.inference_threads, 1)

    def test_deltanet_batch_preserves_state_and_chunk_boundaries(self):
        h, nk, nv, dk, dv = 16, 2, 4, 8, 8
        c = 2 * nk * dk + nv * dv
        def random(shape):
            return (self.rng.standard_normal(shape) * .1).astype(np.float16)
        weights = dict(qkv=random((c, h)), gate=random((nv*dv, h)),
                       out=random((h, nv*dv)), beta=random((nv, h)),
                       alpha=random((nv, h)), conv1d=random((c, 4)),
                       dt_bias=random(nv), a=-np.ones(nv, np.float16),
                       norm=np.ones(dv, np.float16))
        def layer():
            return GatedDeltaNetLayer(0, weights, h, nv, nk, dk, dv, 4,
                                      projection_device="CPU")
        reference, batched = layer(), layer()
        for count in (8, 1, 7, 1):
            x = random((count, h)).astype(np.float32)
            expected = np.stack([reference.forward(row) for row in x])
            actual = batched.forward_sequence(x)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=.005)
            np.testing.assert_allclose(batched.state, reference.state, atol=1e-6, rtol=.005)
            np.testing.assert_allclose(batched.conv_state, reference.conv_state, atol=1e-6, rtol=.005)

    def test_native_auto_fallback_and_required_errors(self):
        with patch("flash_moe.native_expert.NativeExpert", side_effect=OSError("missing")):
            self.assertIsNone(load_native_expert("auto", None, "CPU", {0: dict(gate=11, up=11, down=11)}))
            with self.assertRaises(RuntimeError):
                load_native_expert("native", None, "CPU", {0: dict(gate=11, up=11, down=11)})
        self.assertIsNone(load_native_expert("auto", None, "GPU", {}))
        with self.assertRaises(RuntimeError):
            load_native_expert("native", None, "CPU", {0: dict(gate=999, up=11, down=11)})

    def test_prefill_decode_transition_and_reset(self):
        events = []
        model = object.__new__(Qwen3MoEModel)
        model.backend = SimpleNamespace(begin_prefill=lambda: events.append("prefill"),
                                         begin_decode=lambda: events.append("decode"))
        model.blocks = [SimpleNamespace(is_ssm=True, forward=lambda x: x+1)]
        model.block_count = 1
        model.ssm = {0: SimpleNamespace(reset=lambda: events.append("reset"))}
        for count in (4, 1):
            model.init_caches()
            x = np.zeros((1, count, 16), np.float32)
            np.testing.assert_array_equal(model._forward_all(x), x+1)
            model._forward_all(x[:, :1])
        self.assertEqual(events, ["prefill", "reset", "decode"] * 2)

    @unittest.skipUnless(NATIVE_PATH.is_file(), "compiled native kernel not configured")
    def test_native_deltanet_state_over_changing_inputs(self):
        native = NativeExpert(NATIVE_PATH).delta
        if native is None:
            self.skipTest("rebuild native DLL for DeltaNet checks")
        for nv, dv, dk in ((32, 128, 128), (3, 7, 9)):
            state = self.rng.standard_normal((nv, dv, dk)).astype(np.float32) * .01
            reference = SimpleNamespace(state=state.copy(), head_v_dim=dv,
                                        _kv_buf=np.empty(max(dk, dv), np.float32))
            actual = state.copy()
            for _ in range(40):
                q = self.rng.standard_normal((nv, dk)).astype(np.float32) * .05
                k = self.rng.standard_normal((nv, dk)).astype(np.float32) * .05
                v = self.rng.standard_normal((nv, dv)).astype(np.float32) * .1
                beta = self.rng.uniform(.01, .99, nv).astype(np.float32)
                gate = -self.rng.uniform(.001, 5, nv).astype(np.float32)
                want = np.stack([GatedDeltaNetLayer._delta_step(
                    reference, reference.state[h], q[h], k[h], v[h],
                    float(beta[h]), float(gate[h])) for h in range(nv)])
                got = native.forward(actual, q, k, v, beta, gate)
                np.testing.assert_allclose(got, want, atol=1e-7, rtol=.0001)
                np.testing.assert_allclose(actual, reference.state, atol=1e-7, rtol=.0001)
        # The model selects the native recurrence for decode only and always
        # clears the temporary layer reference, including after failures.
        layer = SimpleNamespace(native_delta=None)
        observed = []
        def sequence(values):
            observed.append(layer.native_delta)
            return values
        layer.forward_sequence = sequence
        model = SimpleNamespace(backend=SimpleNamespace(
            native_expert=SimpleNamespace(delta=native), _native_enabled=False))
        x = np.zeros((1, 3, 16), np.float32)
        Qwen3MoEModel._run_ssm(model, layer, x)
        model.backend._native_enabled = True
        Qwen3MoEModel._run_ssm(model, layer, x[:, :1])
        self.assertEqual(observed, [None, native])
        self.assertIsNone(layer.native_delta)

    @unittest.skipUnless(NATIVE_PATH.is_file(), "compiled native kernel not configured")
    def test_native_quantizations_and_concurrency(self):
        import gguf
        native = NativeExpert(NATIVE_PATH)
        block = {8: (32, 34), 11: (256, 110), 12: (256, 144),
                 13: (256, 176), 14: (256, 210)}
        for kinds in ((8,11,12), (11,12,13), (12,13,14), (13,14,8), (14,8,11)):
            tensors, weights = {}, []
            for name, kind in zip(("gate", "up", "down"), kinds):
                bs, size = block[kind]
                raw = self.rng.integers(0, 256, (256*(256//bs), size), np.uint8)
                offset = size-2 if kind in (11,14) else 0
                raw[:, offset:offset+2] = np.frombuffer(np.float16(.002).tobytes(), np.uint8)
                if kind in (12,13):
                    raw[:, 2:4] = np.frombuffer(np.float16(.001).tobytes(), np.uint8)
                raw = raw.reshape(256, -1)
                tensors[name] = raw.tobytes()
                weights.append(gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(kind)))
            round16 = lambda a: a.astype(np.float16).astype(np.float32)
            weights = list(map(round16, weights))
            x = round16((self.rng.standard_normal(256)*.005).astype(np.float32))
            g, u = round16(x @ weights[0].T), round16(x @ weights[1].T)
            activation = round16(round16(g * round16(1/(1+np.exp(-g)))) * u)
            expected = round16(activation @ weights[2].T)
            types = dict(zip(("gate", "up", "down"), kinds))
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(lambda _: native.forward(tensors, types, x, 256, 256), range(8)))
            error = np.max(np.abs(results[0]-expected))/max(1e-12, np.max(np.abs(expected)))
            self.assertLess(error, .005)
            for got in results:
                np.testing.assert_array_equal(got, results[0])
            with self.assertRaisesRegex(RuntimeError, "status -2"):
                native.forward(tensors, dict(types, gate=999), x, 256, 256)

            # Exercise the real backend switch, raw store, worker reduction,
            # and empty converted cache together, not just the FFI in isolation.
            if kinds == (11,12,13):
                records = [pack_expert(self.root, ExpertKey(0, i), tensors)
                           for i in range(4)]
                manifest = ExpertManifest.load(write_manifest(self.root, records))
                store = AsyncExpertStore(self.root, capacity=4, manifest=manifest)
                self.addCleanup(store.close)
                backend = OpenVINOExpertBackend(
                    store, device="CPU", cache_capacity=1,
                    tensor_types={0: types}, native_decode="native",
                    native_library=NATIVE_PATH,
                )
                backend.core.set_property("CPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32})
                self.addCleanup(backend.close)
                router = synthetic_router(256, 4, 4, seed=1)
                reference = run_moe_layer(0, x, router, backend, 256, 256)
                self.assertEqual(backend.cache_size(), 1)
                backend.begin_decode()
                actual = run_moe_layer(0, x, router, backend, 256, 256)
                error = np.max(np.abs(actual-reference))/max(1e-12, np.max(np.abs(reference)))
                self.assertLess(error, .005)
                self.assertEqual(backend.cache_size(), 0)
                backend.begin_prefill()
                self.assertFalse(backend._native_enabled)


if __name__ == "__main__":
    unittest.main()
