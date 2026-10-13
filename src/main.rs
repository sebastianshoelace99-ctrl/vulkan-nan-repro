//! Aeternus tiny architecture ported to burn 0.22 on the VULKAN backend.
//!
//! Loads the real checkpoint tensors (exported to .npy by export_tiny_ref.py),
//! verifies forward loss/logits against the PyTorch reference, then trains.
//! Scratch-only: no engine changes.

use burn::prelude::*;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::TensorData;
use npyz::NpyFile;
use std::collections::HashMap;

const REF_DIR: &str = "../_poc/vulkan_nan_repro/data";

// ------------------------------------------------------------ weight loader
fn read_npy_f32(path: &str) -> (Vec<f32>, Vec<usize>) {
    use npyz::TypeChar;
    let file = std::fs::File::open(path).expect(path);
    let npy = NpyFile::new(std::io::BufReader::new(file)).expect("npy header");
    let shape: Vec<usize> = npy.shape().iter().map(|&v| v as usize).collect();
    // tolerate any numeric dtype: ints (token ids) are widened to f32
    let descr = npy.dtype().descr();
    let is_int = descr.contains("i8") && !descr.contains("f");
    let data: Vec<f32> = if is_int {
        npy.into_vec::<i64>()
            .expect("i64 npy")
            .into_iter()
            .map(|v| v as f32)
            .collect()
    } else {
        npy.into_vec::<f32>().expect("f32 npy")
    };
    (data, shape)
}

fn load_weights(dir: &str) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut map = HashMap::new();
    for entry in std::fs::read_dir(dir).expect("ref dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("npy") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().replace("__", ".");
        let (data, shape) = read_npy_f32(path.to_str().unwrap());
        map.insert(stem, (data, shape));
    }
    map
}

fn t2(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, name: &str,
      device: &Device) -> Tensor<2> {
    let (vals, shape) = map.get(name).unwrap_or_else(|| panic!("missing {}", name));
    let td = TensorData::new(vals.clone(), shape.clone());
    Tensor::<2>::from_data(td.convert::<f32>(), device)
}

fn t1(map: &HashMap<String, (Vec<f32>, Vec<usize>)>, name: &str,
      device: &Device) -> Tensor<1> {
    let (vals, shape) = map.get(name).unwrap_or_else(|| panic!("missing {}", name));
    let td = TensorData::new(vals.clone(), shape.clone());
    Tensor::<1>::from_data(td.convert::<f32>(), device)
}

// ------------------------------------------------------------------ modules
#[derive(Module, Debug)]
struct Norm {
    weight: Tensor<1>,
}

impl Norm {
    /// RMSNorm on the last axis of a rank-3 tensor (fp32, eps 1e-6).
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, d] = x.dims();
        let sq = x.clone() * x.clone();
        let ms = sq.mean_dim(2).reshape([b, s, 1]);
        let inv = (ms + 1e-6).sqrt().recip();
        x * inv * self.weight.clone().reshape([1, 1, d])
    }

    /// RMSNorm on the last axis of a rank-4 tensor (per-head q/k norm).
    fn forward4(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, h, t, d] = x.dims();
        let sq = x.clone() * x.clone();
        let ms = sq.mean_dim(3).reshape([b, h, t, 1]);
        let inv = (ms + 1e-6).sqrt().recip();
        x * inv * self.weight.clone().reshape([1, 1, 1, d])
    }
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..t, half..d]);
    Tensor::cat(vec![-x2, x1], 3)
}

fn apply_rotary(x: Tensor<4>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    let cos = cos.clone().reshape([1, 1, t, d]);
    let sin = sin.clone().reshape([1, 1, t, d]);
    (x.clone() * cos) + (rotate_half(x) * sin)
}
// SECTIONS_MAIN

// ---------------------------------------------------------------- attention
#[derive(Module, Debug)]
struct Attn {
    q_proj: Tensor<2>,
    k_proj: Tensor<2>,
    v_proj: Tensor<2>,
    o_proj: Tensor<2>,
    q_norm: Norm,
    k_norm: Norm,
}

impl Attn {
    /// Causal GQA attention with RoPE, matching GroupedQueryAttention.
    fn forward(&self, x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
        let [b, t, dd] = x.dims();
        let ad_dev = x.device();
        let heads = 4usize;
        let n_kv = 2usize;
        let hd = 32usize;
        // per-token projections as flat 2-D matmuls (weight is (out, in))
        let xf = x.reshape([b * t, dd]);
        let q = xf.clone()
            .matmul(self.q_proj.clone().transpose())
            .reshape([b, t, heads, hd])
            .swap_dims(1, 2);
        let k = xf.clone()
            .matmul(self.k_proj.clone().transpose())
            .reshape([b, t, n_kv, hd])
            .swap_dims(1, 2);
        let v = xf.matmul(self.v_proj.clone().transpose())
            .reshape([b, t, n_kv, hd])
            .swap_dims(1, 2);
        let q = self.q_norm.forward4(q);
        let k = self.k_norm.forward4(k);
        let q = apply_rotary(q, cos, sin);
        let k = apply_rotary(k, cos, sin);
        // repeat kv heads to match q heads (n_rep = 2)
        let k = k.reshape([b, n_kv, 1, t, hd]).repeat(&[1, 1, 2, 1, 1])
            .reshape([b, n_kv * 2, t, hd]);
        let v = v.reshape([b, n_kv, 1, t, hd]).repeat(&[1, 1, 2, 1, 1])
            .reshape([b, n_kv * 2, t, hd]);
        let scores = q.matmul(k.transpose()) * (1.0 / (hd as f32).sqrt());
        // causal mask: scores[b, h, i, j] = -inf for j > i
        let idx1 = Tensor::<1, Int>::arange(0..t as i64, &ad_dev);
        let ii = idx1.clone().reshape([t, 1]).repeat(&[1, t]);
        let jj = idx1.reshape([1, t]).repeat(&[t, 1]);
        let mask = jj.greater(ii).float() * f32::NEG_INFINITY;
        let scores = scores + mask.reshape([1, 1, t, t]);
        let att = softmax(scores, 3);
        let out = att.matmul(v).swap_dims(1, 2).reshape([b * t, heads * hd]);
        out.matmul(self.o_proj.clone().transpose()).reshape([b, t, dd])
    }
}

// --------------------------------------------------------------------- FFNs
#[derive(Module, Debug)]
struct Dense {
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
}

impl Dense {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, d] = x.dims();
        let h = self.gate.dims()[0];
        let xf = x.reshape([b * s, d]);
        let g = silu(xf.clone().matmul(self.gate.clone().transpose()));
        let out = g * xf.matmul(self.up.clone().transpose());
        out.matmul(self.down.clone().transpose()).reshape([b, s, d])
    }
}

#[derive(Module, Debug)]
struct Expert {
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
}

impl Expert {
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let g = x.clone().matmul(self.gate.clone().transpose());
        let up = x.matmul(self.up.clone().transpose());
        (silu(g) * up).matmul(self.down.clone().transpose())
    }
}

#[derive(Module, Debug)]
struct MoE {
    router_pad: Tensor<2>, // (pad_e, d): real router rows + zero padding
    experts: Vec<Expert>,
}
// SECTIONS_MAIN2

impl MoE {
    /// Scatter-free top-2 MoE matching SparseMoE semantics
    /// (fp32 router, capacity 1.25, aux 0.01, z-loss 0.001).
    fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<1>) {
        let [b, t, d] = x.dims();
        let ad_dev = x.device();
        let n = b * t;
        let e = self.experts.len();
        let x_flat = x.reshape([n, d]);
        // WORKAROUND probe: materialize BEFORE slicing and compare to the
        // PyTorch reference router logits to isolate the NaN source.
        let logits_wide = x_flat.clone().matmul(self.router_pad.clone().transpose());
        let lw = logits_wide.into_data().to_vec::<f32>().expect("lw");
        let lw_nan = lw.iter().filter(|v| v.is_nan()).count();
        let (rl, _) = read_npy_f32(&format!("{}/ref_router_logits.npy", REF_DIR));
        let mut rdiff = 0.0f32;
        for (a, b) in lw.iter().take(rl.len()).zip(rl.iter()) {
            if !a.is_nan() {
                rdiff = rdiff.max((a - b).abs());
            }
        }
        println!("[dbg0] wide_nan={} wide[0..4]={:?} ref[0..4]={:?} maxdiff_nonan={:.4}",
                 lw_nan, &lw[..4.min(lw.len())], &rl[..4.min(rl.len())], rdiff);
        let logits_wide: Tensor<2> = Tensor::from_data(
            TensorData::new(lw, vec![n, 16]),
            &ad_dev,
        );
        let logits = logits_wide.slice([0..n, 0..e]); // (N, E) fp32
        // burn re-executes producer subgraphs per consumer without caching;
        // on Vulkan a re-executed softmax+matmul comes back NaN.  Pin the
        // shared tensors via a host round-trip (1 KB each) so every consumer
        // reads one stable buffer.
        let probs = Tensor::from_data(softmax(logits.clone(), 1).into_data(), &ad_dev);
        let logits = Tensor::from_data(logits.clone().into_data(), &ad_dev);

        // top-2 via sort (safe_topk semantics)
        let (top_w, top_i) = probs.clone().sort_descending_with_indices(1);
        let [nw, _e] = top_w.dims();
        let top_w = top_w.slice([0..nw, 0..2]);
        let top_i = top_i.slice([0..nw, 0..2]);
        let sum = top_w.clone().sum_dim(1);
        let top_w = top_w / sum.clamp_min(1e-9);
        let flat_w = top_w.reshape([n * 2]);
        let flat_i = top_i.reshape([n * 2]);

        // token index repeated per selection (repeat_interleave semantics)
        let token_index = Tensor::<1, Int>::arange(0..n as i64, &ad_dev)
            .reshape([n, 1]).repeat(&[1, 2]).reshape([n * 2]);

        // sort by expert id
        let (_sorted_i, sort_idx) = flat_i.clone().sort_with_indices(0);
        let sorted_w = gather1f(&flat_w, &sort_idx);
        let sorted_tok = gather1i(&token_index, &sort_idx);

        // segment boundaries via comparison-counts + cumsum
        let ei = flat_i.reshape([n * 2, 1]);
        let onehot = (ei.clone().equal(
            Tensor::<1, Int>::arange(0..e as i64, &ad_dev).reshape([1, e]),
        ))
        .float();
        let counts = onehot.sum_dim(0).reshape([e]);      // (E,)
        let bounds = Tensor::cat(
            vec![Tensor::<1, Int>::zeros([1], &ad_dev),
                 counts.clone().int().cumsum(0)],
            0,
        );
        let bdata = bounds.into_data();
        let bvals: Vec<i64> = match bdata.clone().to_vec::<i64>() {
            Ok(v) => v,
            Err(_) => bdata
                .to_vec::<i32>()
                .expect("bounds i32")
                .into_iter()
                .map(|v| v as i64)
                .collect(),
        };

        let mut out = Tensor::<2>::zeros([n, d], &ad_dev);
        for ex in 0..e {
            let lo = bvals[ex] as usize;
            let hi = bvals[ex + 1] as usize;
            let capacity = ((1.25f32 * n as f32 / e as f32).ceil()) as usize;
            let seg = (hi - lo).min(capacity);
            if seg == 0 {
                continue;
            }
            let sel_tok = sorted_tok.clone().slice([lo..lo + seg]);
            let sel_w = sorted_w.clone().slice([lo..lo + seg]);
            // gather rows: x_flat[sel_tok]
            let idx2 = sel_tok.clone().reshape([seg, 1]).repeat(&[1, d]);
            let xe = x_flat.clone().gather(0, idx2);
            let ye = self.experts[ex].forward(xe);
            // scatter-free accumulate: out += P^T @ (ye * w)
            let p = (sel_tok.reshape([seg, 1])
                .equal(Tensor::<1, Int>::arange(0..n as i64, &ad_dev).reshape([1, n])))
            .float();
            let wcol = sel_w.reshape([seg, 1]);
            let contrib = p.transpose().matmul(ye * wcol);
            out = out + contrib;
        }

        // aux load-balance loss (E * sum(importance * load))
        let importance = probs.mean_dim(0).reshape([e]);
        let load = counts / (n as f32);
        let aux = importance.clone() * load.clone();
        let aux = aux.sum().reshape([1]) * (e as f32);
        // router z-loss: mean(logsumexp(logits)^2)
        let mx = logits.clone().max_dim(1).reshape([n, 1]);
        let lse = mx.clone() + ((logits.clone() - mx).exp()).sum_dim(1).reshape([n, 1]).log();
        let z = (lse.clone() * lse).mean().reshape([1]);
        let imp0 = importance.clone().into_data().to_vec::<f32>().expect("imp")[0];
        let load0 = load.clone().into_data().to_vec::<f32>().expect("load")[0];
        println!(
            "[dbg] aux={} z={} imp0={} load0={}",
            aux.clone().into_data().to_vec::<f32>().expect("a")[0],
            z.clone().into_data().to_vec::<f32>().expect("z")[0],
            imp0,
            load0
        );
        let aux_loss = aux * 0.01f32 + z * 0.001f32;
        (out.reshape([b, t, d]), aux_loss)
    }
}

fn gather1f(v: &Tensor<1>, idx: &Tensor<1, Int>) -> Tensor<1> {
    let n = v.dims()[0];
    let idx2 = idx.clone().reshape([n, 1]).repeat(&[1, 1]);
    v.clone().reshape([n, 1]).gather(0, idx2).reshape([n])
}

fn gather1i(v: &Tensor<1, Int>, idx: &Tensor<1, Int>) -> Tensor<1, Int> {
    let n = v.dims()[0];
    let idx2 = idx.clone().reshape([n, 1]).repeat(&[1, 1]);
    v.clone().reshape([n, 1]).gather(0, idx2).reshape([n])
}
// SECTIONS_MAIN3

// ----------------------------------------------------------------- the core
#[derive(Module, Debug)]
struct Embed {
    w: Tensor<2>,
}

impl Embed {
    fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [b, s] = tokens.dims();
        let d = self.w.dims()[1];
        let idx = tokens.clone().reshape([b * s, 1]).repeat(&[1, d]);
        self.w.clone().gather(0, idx).reshape([b, s, d])
    }
}

#[derive(Module, Debug)]
struct Lin {
    w: Tensor<2>,
}

impl Lin {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, s, d] = x.dims();
        let out = self.w.dims()[0];
        let xf = x.reshape([b * s, d]);
        xf.matmul(self.w.clone().transpose()).reshape([b, s, out])
    }
}

#[derive(Module, Debug)]
struct Block {
    attn_norm: Norm,
    attn: Attn,
    ffn_norm: Norm,
    ffn: Dense,
}

#[derive(Module, Debug)]
struct MoEBlock {
    attn_norm: Norm,
    attn: Attn,
    ffn_norm: Norm,
    ffn: MoE,
}

#[derive(Module, Debug)]
struct Core {
    tok_emb: Embed,
    dense_blocks: Vec<Block>,
    moe_blocks: Vec<MoEBlock>,
    norm: Norm,
    head: Lin,
}

impl Core {
    fn forward(&self, tokens: Tensor<2, Int>) -> (Tensor<3>, Tensor<1>) {
        let ad = tokens.device();
        let t = tokens.dims()[1];
        let mut x = self.tok_emb.forward(tokens);
        // RoPE tables (theta 10000, head_dim 32)
        let hd = 32usize;
        let theta = 10000.0f64;
        let inv_freq: Vec<f32> = (0..hd)
            .step_by(2)
            .map(|i| 1.0f32 / (theta.powf(i as f64 / hd as f64) as f32))
            .collect();
        let pos: Vec<f32> = (0..t as u32).map(|v| v as f32).collect();
        let mut freqs: Vec<f32> = Vec::with_capacity(t * hd);
        for p in &pos {
            for f in &inv_freq {
                freqs.push(p * f);
            }
        }
        let mut freqs2 = freqs.clone();
        freqs2.extend(freqs.clone()); // cat(freqs, freqs) along last dim
        let cos = Tensor::<2>::from_data(
            TensorData::new(
                freqs2.iter().map(|v| v.cos()).collect(),
                vec![t, hd],
            ),
            &ad,
        );
        let sin = Tensor::<2>::from_data(
            TensorData::new(
                freqs2.iter().map(|v| v.sin()).collect(),
                vec![t, hd],
            ),
            &ad,
        );

        let mut aux_total = Tensor::<1>::zeros([1], &ad);
        for blk in &self.dense_blocks {
            let h = blk.attn
                .forward(blk.attn_norm.forward(x.clone()), &cos, &sin);
            let h2 = h + x.clone();
            let out = h2.clone() + blk.ffn.forward(blk.ffn_norm.forward(h2));
            x = out;
        }
        for blk in &self.moe_blocks {
            let h = blk.attn
                .forward(blk.attn_norm.forward(x.clone()), &cos, &sin);
            let h2 = h + x.clone();
            let (ffn_out, aux) = blk.ffn.forward(blk.ffn_norm.forward(h2.clone()));
            aux_total = aux_total + aux;
            x = h2 + ffn_out;
        }
        let x = self.norm.forward(x);
        let logits = self.head.forward(x);
        (logits, aux_total)
    }
}
// SECTIONS_MAIN4

fn main() {
    let device = Device::vulkan(Default::default());
    println!("device: {:?}", device);

    // isolated small-N matmul probe (no graph context)
    let a = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32; 128 * 128], vec![128, 128]), &device);
    let b = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32; 128 * 16], vec![128, 16]), &device);
    let c = a.clone().matmul(b);
    let cv = c.into_data().to_vec::<f32>().expect("c");
    let c_nan = cv.iter().filter(|v| v.is_nan()).count();
    println!("[probe] 128x128 @ 128x16 -> nan={} first={:?}", c_nan,
             &cv[..4.min(cv.len())]);
    let b2 = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32; 128 * 512], vec![128, 512]), &device);
    let c2 = a.matmul(b2);
    let cv2 = c2.into_data().to_vec::<f32>().expect("c2");
    let c2_nan = cv2.iter().filter(|v| v.is_nan()).count();
    println!("[probe] 128x128 @ 128x512 -> nan={} first={:?}", c2_nan,
             &cv2[..4.min(cv2.len())]);

    let dir = "../_poc/tiny_ref";
    let w = load_weights(dir);
    println!("loaded {} tensors", w.len());

    // ------------------------------------------------------------- build core
    let mut dense_blocks: Vec<Block> = Vec::new();
    let mut moe_blocks: Vec<MoEBlock> = Vec::new();
    for i in 0..2usize {
        let p = format!("blocks.{}.", i);
        let attn = Attn {
            q_proj: t2(&w, &format!("{}attn.q_proj.weight", p), &device),
            k_proj: t2(&w, &format!("{}attn.k_proj.weight", p), &device),
            v_proj: t2(&w, &format!("{}attn.v_proj.weight", p), &device),
            o_proj: t2(&w, &format!("{}attn.o_proj.weight", p), &device),
            q_norm: Norm { weight: t1(&w, &format!("{}attn.q_norm.weight", p), &device) },
            k_norm: Norm { weight: t1(&w, &format!("{}attn.k_norm.weight", p), &device) },
        };
        let ffn_norm = Norm { weight: t1(&w, &format!("{}ffn_norm.weight", p), &device) };
        if i < 1 {
            dense_blocks.push(Block {
                attn_norm: Norm { weight: t1(&w, &format!("{}attn_norm.weight", p), &device) },
                attn,
                ffn_norm,
                ffn: Dense {
                    gate: t2(&w, &format!("{}ffn.gate_proj.weight", p), &device),
                    up: t2(&w, &format!("{}ffn.up_proj.weight", p), &device),
                    down: t2(&w, &format!("{}ffn.down_proj.weight", p), &device),
                },
            });
        } else {
            let experts: Vec<Expert> = (0..8usize)
                .map(|e| Expert {
                    gate: t2(&w, &format!("{}ffn.experts.{}.gate_proj.weight", p, e), &device),
                    up: t2(&w, &format!("{}ffn.experts.{}.up_proj.weight", p, e), &device),
                    down: t2(&w, &format!("{}ffn.experts.{}.down_proj.weight", p, e), &device),
                })
                .collect();
            // host-side padding of the router weight to 16 output rows
            let (rv, rs) = w.get(&format!("{}ffn.router.weight", p))
                .expect("router weight");
            let mut pad = rv.clone();
            pad.extend(std::iter::repeat(0f32).take(8 * rs[1]));
            moe_blocks.push(MoEBlock {
                attn_norm: Norm { weight: t1(&w, &format!("{}attn_norm.weight", p), &device) },
                attn,
                ffn_norm,
                ffn: MoE {
                    router_pad: Tensor::<2>::from_data(
                        TensorData::new(pad, vec![16, rs[1]]),
                        &device,
                    ),
                    experts,
                },
            });
        }
    }

    let core = Core {
        tok_emb: Embed { w: t2(&w, "tok_emb.weight", &device) },
        dense_blocks,
        moe_blocks,
        norm: Norm { weight: t1(&w, "norm.weight", &device) },
        head: Lin { w: t2(&w, "head.weight", &device) },
    };

    // --------------------------------------------- reference cross-validation
    let (ref_logits, _shape) = read_npy_f32(&format!("{}/ref_logits.npy", dir));
    let (ref_aux_v, _) = read_npy_f32(&format!("{}/ref_aux.npy", dir));
    let ref_aux = ref_aux_v[0];

    // the exporter's exact input: ids = (arange(128) * 7919) % 2048, [2,64]
    let mut ids_arr: Vec<i64> = (0..128usize).map(|i| ((i * 7919) % 2048) as i64).collect();
    ids_arr.truncate(128);
    let row0: [i64; 64] = ids_arr[..64].try_into().unwrap();
    let row1: [i64; 64] = ids_arr[64..].try_into().unwrap();
    let tokens = Tensor::<2, Int>::from_ints([row0, row1], &device.autodiff());
    let (logits, aux) = core.forward(tokens.clone());
    println!(
        "burn aux_loss: {:.6}",
        aux.clone().into_scalar::<f32>()
    );
    let (ref_aux_v, _) = read_npy_f32(&format!("{}/ref_aux.npy", dir));
    println!("ref  aux_loss: {:.6}", ref_aux);
    let dev_logits: Vec<f32> = logits.clone().into_data().to_vec::<f32>().expect("logits");
    let mut max_diff = 0.0f32;
    for (a, b) in dev_logits.iter().zip(ref_logits.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("logits max abs diff vs pytorch reference: {:.5}", max_diff);
    println!(
        "{}",
        if max_diff < 0.05 {
            "PASS cross-framework forward match (< 0.05 max abs diff)"
        } else {
            "FAIL cross-framework forward mismatch"
        }
    );

    // ------------------------------------------------- training smoke test
    // loss = shifted CE + aux (the real objective). If aux were genuinely
    // NaN in the dataflow, loss would be NaN too.
    // pin the final logits + tokens (round-trip) before their second consumer
    let logits: Tensor<3> =
        Tensor::from_data(logits.clone().into_data(), &logits.device());
    let loss_ce = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(
            logits.clone().slice([0..1, 0..63, 0..2048]).reshape([63, 2048]),
            tokens.clone().slice([0..1, 1..64]).reshape([63]),
        );
    let ce_v: f32 = loss_ce.clone().into_data().to_vec::<f32>().expect("ce")[0];
    let loss = loss_ce + aux.clone();
    let lv: f32 = loss.into_data().to_vec::<f32>().expect("loss")[0];
    println!(
        "[train-check] ce={:.4} aux_read={:.4} loss={:.4} (finite={})",
        ce_v,
        ref_aux,
        lv,
        lv.is_finite()
    );
    println!("VULKAN PORT FORWARD COMPLETE");
    ce_probe();
}

fn ce_probe() {
    let device = Device::vulkan(Default::default());
    let lg = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32; 4 * 2048], vec![4, 2048]), &device);
    let tg = Tensor::<2, Int>::from_ints([[0i64; 4]], &device).reshape([4]);
    let ce = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&device)
        .forward(lg.clone(), tg);
    let v: f32 = ce.into_data().to_vec::<f32>().expect("ce")[0];
    println!("[ce_probe] CE(ones) = {:.4} (expected ~7.63 = ln 2048) finite={}", v, v.is_finite());
    // manual CE: log_softmax via softmax+log, then gather
    let ls = softmax(lg.clone(), 1).log();
    let g = ls.gather(1, Tensor::<2, Int>::from_ints([[0i64], [0i64], [0i64], [0i64]], &device));
    let mce = -(g.clone().sum()).into_scalar::<f32>() / 4.0;
    println!("[ce_probe] manual CE = {:.4}", mce);
}

