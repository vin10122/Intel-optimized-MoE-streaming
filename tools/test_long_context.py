"""Independent GQA and KV regression checks; no GGUF or GPU required."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from flash_moe.attention import grouped_attention, KVCache


class LongContextTests(unittest.TestCase):
    def test_grouped_matmul_matches_replicated_reference(self):
        rng = np.random.default_rng(31)
        cases = ((1,1155,16,2,256,1154), (1,1567,16,2,256,1566),
                 (17,17,8,2,32,0), (7,23,6,2,16,16), (4,4,2,2,8,0))
        for nq,nk,nh,nkv,dim,offset in cases:
            with self.subTest(shape=(nq,nk,nh,nkv,dim)):
                q=rng.standard_normal((nq,nh,dim)).astype(np.float32)
                k=rng.standard_normal((nk,nkv,dim)).astype(np.float16)
                v=rng.standard_normal((nk,nkv,dim)).astype(np.float16)
                # Independent historical formulation with replicated KV heads.
                kr=np.repeat(k.astype(np.float32),nh//nkv,axis=1)
                vr=np.repeat(v.astype(np.float32),nh//nkv,axis=1)
                scores=np.einsum('qhd,khd->qhk',q,kr)*(1./np.sqrt(dim))
                mask=np.arange(nk)[None,:] > (offset+np.arange(nq))[:,None]
                scores=np.where(mask[:,None,:],-np.inf,scores)
                scores-=scores.max(axis=-1,keepdims=True)
                p=np.exp(scores);p/=p.sum(axis=-1,keepdims=True)
                expected=np.einsum('qhk,khd->qhd',p,vr).reshape(nq,nh*dim)
                actual=grouped_attention(q,k,v,nq,offset)
                self.assertEqual(actual.dtype,np.float32)
                np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)

    def test_kv_storage_is_reused_with_exact_rounding(self):
        rng=np.random.default_rng(7)
        cache=KVCache(2,16,max_seq_len=128)
        key_pointer=cache._key_storage.ctypes.data
        value_pointer=cache._value_storage.ctypes.data
        keys=[];values=[]
        for count in (7,1,13,1,32):
            k=rng.standard_normal((count,2,16)).astype(np.float32)
            v=rng.standard_normal((count,2,16)).astype(np.float32)
            keys.append(k.astype(np.float16));values.append(v.astype(np.float16))
            cache.append(k,v)
            a,b=cache.get()
            np.testing.assert_array_equal(a,np.concatenate(keys))
            np.testing.assert_array_equal(b,np.concatenate(values))
            self.assertEqual(cache._key_storage.ctypes.data,key_pointer)
            self.assertEqual(cache._value_storage.ctypes.data,value_pointer)
        before=[a.copy() for a in cache.get()]
        with self.assertRaises(ValueError):
            cache.append(np.zeros((100,2,16)),np.zeros((100,2,16)))
        for a,b in zip(before,cache.get()):np.testing.assert_array_equal(a,b)
        cache.reset()
        self.assertEqual(cache.length,0)
        self.assertEqual(cache.get()[0].shape,(0,2,16))


if __name__=='__main__':
    unittest.main()
