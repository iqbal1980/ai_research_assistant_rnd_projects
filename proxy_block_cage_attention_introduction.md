# Proxy Block-CAGE: A Sparse Attention Experiment from an RTX 2080 Laptop

I started this project with a question that was half technical and half personal:

> Could I use ChatGPT as a serious research partner, not just as a code generator, to help me explore a core AIML research problem?

I am **Iqbal Addou**, a **PhD student in Bioinformatics and Computational Biology at George Mason University**. I have about **23 years of software engineering experience** and about **5 years of applied AIML engineering experience**, mostly in computer vision. I know PyTorch, training workflows, and how to use modern AI frameworks, but I do not come from a long background in core model research. I am trying to pivot toward that world: efficient inference, sparse attention, long-context systems, and the mathematical/computational side of Transformers.

In my PhD work I have used genetic programming, genetic algorithms, optimization techniques, and physics-informed neural networks. Those methods are not always fashionable in modern LLM work, but I have learned to respect them because they force you to think in terms of search spaces, objectives, constraints, and failure modes. So I wanted to see if some of that mindset could be useful in a Transformer problem.

The first prompt was intentionally broad and a little unusual. I asked ChatGPT to help me find a computationally and mathematically intensive problem in LLMs and Transformers, and to explore out-of-the-box methods such as genetic programming, genetic algorithms, classical AI/ML, and older optimization ideas. I wanted something that could be formulated mathematically, implemented locally, and tested on my own hardware.

The project eventually became **Proxy Block-CAGE**, a small sparse-attention prototype for long-context prefill. The useful version is simple:

> Split Q/K/V into blocks, use cheap pooled query/key block summaries to select a few key blocks per query block, then compute exact local attention only inside those selected key/value blocks.

The strongest result I got was on my laptop GPU: an **NVIDIA RTX 2080 Max-Q 8GB**. In a tiny **2-layer character-level Transformer** trained on a custom text corpus, at **16,384 tokens**, Proxy Block-CAGE ran generation-style prefill **2.61x faster** than dense PyTorch attention, while using only **0.39%** of dense attention capacity and preserving **100% final-token top-1/top-5 agreement** on that small eval run.

This is not a production LLM claim. It is a reproducible research prototype and a request for honest feedback.

---

## The result in one table

The most important runs were on a trained 2-layer character-level Transformer, using CUDA fp16 on the RTX 2080 Max-Q.

| Context length | Dense prefill | Proxy Block-CAGE, capacity 64 | Speedup | Attention density | Final-token top-1 agreement | Final-token top-5 containment |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 7.925 ms | 5.263 ms | 1.51x | 0.781% | 95% | 100% |
| 12,288 | 14.980 ms | 7.018 ms | 2.13x | 0.521% | 100% | 100% |
| 16,384 | 24.301 ms | 9.323 ms | 2.61x | 0.391% | 100% | 100% |

For the 16,384-token run:

```text
dense prefill_last_ms_median:        24.3009 ms
Proxy Block-CAGE 64 prefill median:   9.3229 ms
speedup:                              2.61x
density:                              0.00390625 = 0.390625%
final-token top-1 agreement:          100%
final-token top-5 containment:        100%
```

The model was tiny, the corpus was small, and the eval sample count was small. I am treating this as a stronger sanity check than random tensors, not as evidence that this works for production LLMs.

---

## Why attention is expensive

After projection, a Transformer layer has:

$$
Q,K,V \in \mathbb{R}^{B \times H \times n \times d}
$$

where:

```text
B = batch size
H = number of attention heads
n = sequence length
d = head dimension
```

Dense causal attention computes:

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt d}
$$

for valid causal pairs:

$$
j \le i
$$

Then:

$$
a_{ij} = \frac{\exp(s_{ij})}{\sum_{\ell \le i}\exp(s_{i\ell})}
$$

and:

$$
y_i = \sum_{j \le i} a_{ij}v_j
$$

The compute is approximately:

$$
O(BHn^2d)
$$

At 16,384 tokens, one head has:

$$
\frac{n(n+1)}{2}
= \frac{16384 \cdot 16385}{2}
= 134{,}225{,}920
$$

causal token-token score locations. With 4 heads:

$$
4 \cdot 134{,}225{,}920 = 536{,}903{,}680
$$

That is about **537 million query-key score locations** for one layer.

Dense attention is extremely optimized in PyTorch and FlashAttention-style kernels, so reducing arithmetic does not automatically mean a faster wall-clock runtime. But the arithmetic shows why long context is a natural place to look for sparsity.

---

## The first idea: evolve attention cages

The original CAGE idea was not block routing. It was genetic/evolutionary attention-mask search.

For a query token \(i\), instead of attending to every previous key, define a small selected set:

$$
C_i \subseteq \{0,1,\dots,i\}
$$

Then compute attention only inside that selected set:

$$
y_i^{\text{CAGE}}
=
\sum_{j\in C_i}
\operatorname{softmax}_{j\in C_i}
\left(
\frac{q_i^\top k_j}{\sqrt d}
\right)v_j
$$

In the genetic algorithm version, an individual chromosome was a list of selected key indices. For example:

```text
query token i = 100
candidate cage C_i = [2, 7, 19, 44, 88, 91, 97, 100]
```

That candidate means token 100 is only allowed to attend to those key/value positions.

A simple fitness function was:

$$
f(C_i)
=
\log \sum_{j\in C_i}
\exp\left(\frac{q_i^\top k_j}{\sqrt d}\right)
$$

The GA operations were ordinary evolutionary operations:

```text
selection: keep better cages
elitism: preserve the best cages
crossover: mix two cages
mutation: replace some indices
repair: remove duplicates and enforce causal validity
```

This was useful conceptually and terrible computationally. Token-level CAGE required one GA search per query token. At 512 tokens, batch 1, 4 heads, that means:

$$
1 \cdot 4 \cdot 512 = 2048
$$

separate GA units.

In one early benchmark, at only 512 tokens, token-level CAGE took about **7.29 seconds**, while dense PyTorch attention took about **0.25 ms**. That failure was important. It told me that the idea had to move from token-level search to something coarser.

---

## The second idea: evolve key blocks instead of tokens

The next step was to group the sequence into blocks.

A **key block** is just a fixed-size contiguous group of key vectors after projection. A Transformer forms:

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V
$$

For each token position \(j\), there is a key vector \(k_j\) and a value vector \(v_j\). With:

```text
key_block_size = 32
```

we get:

```text
key block 0 = token positions 0..31
key block 1 = token positions 32..63
key block 2 = token positions 64..95
...
```

Mathematically:

$$
R_s = \{sb_k, sb_k+1, \dots, (s+1)b_k-1\}
$$

where \(b_k\) is the key block size.

A key block is not a paragraph, sentence, document chunk, or learned memory slot. In this prototype it is simply a contiguous range of key/value vectors.

Instead of evolving token cages, the block version evolves:

$$
C_r \subseteq \{0,1,\dots,N_k-1\}
$$

where \(C_r\) is the set of selected key blocks for query block \(r\), and \(N_k\) is the number of key blocks.

Example:

```text
query block = tokens 9600..9631
candidate block cage = [41, 299]
```

If each key block has 32 tokens, selecting 2 key blocks gives each query token at most:

$$
2 \cdot 32 = 64
$$

candidate key/value positions.

This reduced the number of GA searches from one per query token to one per query block. It was a better direction, but it was still too slow in Python.

---

## The mathematical correction: the GA objective collapsed to top-k

The block-level objective I used was approximately:

$$
F(C_r)
=
\log\sum_{s\in C_r}\exp(A_{r,s})
$$

where \(A_{r,s}\) is the score for key block \(s\) with respect to query block \(r\).

For a fixed cage size \(m\), this objective does not need a genetic algorithm. The exact solution is:

$$
C_r^* = \operatorname{TopK}_s(A_{r,s}, m)
$$

The proof is simple. Suppose a selected block \(a\in C_r\) has lower score than an unselected block \(b\notin C_r\):

$$
A_{r,b} > A_{r,a}
$$

Then:

$$
\exp(A_{r,b}) > \exp(A_{r,a})
$$

Replacing \(a\) with \(b\) increases:

$$
\sum_{s\in C_r}\exp(A_{r,s})
$$

and therefore increases:

$$
\log\sum_{s\in C_r}\exp(A_{r,s})
$$

So any optimum must contain the top \(m\) block scores.

This was one of the most useful moments in the project. The genetic algorithm helped me frame the problem, but the math killed the GA for this simple objective.

That changed the story from:

```text
evolve attention cages with GA
```

to:

```text
use the cage formulation, but select blocks with exact top-k
```

---

## The third idea: avoid the full token-score matrix

Exact top-k block selection helped, but the early version still depended on too much full token-pair information. That defeats the purpose.

The next refinement was **Proxy Block-CAGE**.

Instead of scoring all token pairs first, I pool each query block and key block into summaries:

$$
\bar q_r = \frac{1}{|U_r|}\sum_{i\in U_r} q_i
$$

$$
\bar k_s = \frac{1}{|R_s|}\sum_{j\in R_s} k_j
$$

Then score query block \(r\) against key block \(s\):

$$
S_{r,s} = \frac{\bar q_r^\top \bar k_s}{\sqrt d}
$$

Then select:

$$
C_r = \operatorname{TopK}_s(S_{r,s}, m)
$$

where \(m\) is the number of selected key blocks.

In the best default configuration:

```text
query_block_size = 32
key_block_size   = 32
block_cage_size  = 2
selected capacity = 64 tokens
pool             = mean
```

At 16,384 tokens:

$$
N_q = N_k = \frac{16384}{32} = 512
$$

The number of proxy block scores across 4 heads is:

$$
4 \cdot 512 \cdot 512 = 1{,}048{,}576
$$

The local selected token-score upper bound is:

$$
4 \cdot 16384 \cdot 64 = 4{,}194{,}304
$$

So the approximate score locations are:

$$
1{,}048{,}576 + 4{,}194{,}304 = 5{,}242{,}880
$$

Dense causal attention at the same size has about:

$$
536{,}903{,}680
$$

score locations across 4 heads.

The ratio is:

$$
\frac{536{,}903{,}680}{5{,}242{,}880} \approx 102.4
$$

So at 16K context, this configuration touches roughly **100x fewer score locations** than dense causal attention. That does not translate to 100x wall-clock speedup because dense attention kernels are highly optimized and this prototype is not. But it explains why the sparse method begins to win at long context.

---

## Proxy Block-CAGE algorithm

For each layer, batch item, and attention head:

1. Compute \(Q,K,V\).
2. Split query positions into query blocks \(U_r\).
3. Split key/value positions into key blocks \(R_s\).
4. Mean-pool each query block and key block:

   $$
   \bar q_r = \frac{1}{|U_r|}\sum_{i\in U_r}q_i
   $$

   $$
   \bar k_s = \frac{1}{|R_s|}\sum_{j\in R_s}k_j
   $$

5. Compute proxy block scores:

   $$
   S_{r,s}=\frac{\bar q_r^\top \bar k_s}{\sqrt d}
   $$

6. Apply a causal block mask so query blocks cannot route to future-only key blocks.
7. Select the top \(m\) key blocks:

   $$
   C_r = \operatorname{TopK}_s(S_{r,s},m)
   $$

8. Expand selected key blocks into selected key/value token positions.
9. For every query token \(i\in U_r\), compute exact causal attention only over selected positions:

   $$
   J_i = \{j \in \cup_{s\in C_r}R_s : j \le i\}
   $$

   $$
   a_{ij}^{\text{CAGE}}
   =
   \frac{
   \exp\left(q_i^\top k_j / \sqrt d\right)
   }{
   \sum_{\ell\in J_i}
   \exp\left(q_i^\top k_\ell / \sqrt d\right)
   }
   $$

   $$
   y_i^{\text{CAGE}}
   =
   \sum_{j\in J_i} a_{ij}^{\text{CAGE}}v_j
   $$

The approximation is in the selection of which blocks to include. Once blocks are selected, the local attention inside the selected cage is exact softmax attention.

---

## Complexity

Dense attention is:

$$
O(BHn^2d)
$$

Proxy Block-CAGE has two main costs.

First, all block-pair proxy scoring:

$$
O\left(BH d \frac{n^2}{b_qb_k}\right)
$$

Second, exact local attention inside selected blocks:

$$
O(BHnmb_kd)
$$

where:

```text
b_q = query block size
b_k = key block size
m   = number of selected key blocks
```

So the total is approximately:

$$
O\left(
BHd\left[
\frac{n^2}{b_qb_k} + nmb_k
\right]
\right)
$$

With:

```text
b_q = 32
b_k = 32
m   = 2
```

this becomes:

$$
O\left(BHd\left[\frac{n^2}{1024}+64n\right]\right)
$$

This is important: with fixed block size, Proxy Block-CAGE is still technically quadratic because of the all-block-pair proxy score matrix. It is not truly linear attention. It is **block-reduced quadratic routing plus sparse exact local attention**.

That is why I am not claiming this beats systems like SubQ/SSA or other large-scale sparse attention approaches. The current method is a simpler, reproducible prototype with a clear speed/quality tradeoff on consumer hardware.

---

## The prompting loop that mattered

The useful part of ChatGPT was not one magic prompt. It was the iterative loop.

The pattern was:

```text
1. Formulate a strange research direction.
2. Turn it into math.
3. Implement a runnable PyTorch prototype.
4. Run it locally on my RTX 2080.
5. Paste the benchmark output back.
6. Interpret the result honestly.
7. Kill or simplify what failed.
8. Design the next benchmark.
```

The most important instruction I kept giving was essentially:

> Do not just defend the idea. Tell me what the benchmark proves, what it falsifies, and what to try next.

That mattered because many things failed:

```text
Token-level GA attention: too slow.
Block-level GA attention: still too slow.
Exact full-score top-k: cleaner, but too memory-heavy.
Proxy block selection: finally promising.
Training/backward: not ready.
GP local kernels: mathematically interesting, not faster in PyTorch yet.
```

The process felt less like asking for answers and more like having a tireless research engineer who could help me keep reformulating the problem.

---

## The experiment progression

### 1. Token-level genetic attention failed

The first prototype evolved a cage of key tokens for every query token. It worked conceptually, but it was absurdly slow.

At 512 tokens:

```text
dense PyTorch attention: ~0.25 ms
token-level CAGE:       ~7289 ms
```

This killed token-level GA as a practical implementation.

### 2. Block-level GA reduced the search count but was still too slow

Moving from tokens to blocks helped because the number of GA searches dropped from one per query token to one per query block.

But the Python/GA implementation was still seconds at modest context lengths.

### 3. Top-k replaced GA for the simple block objective

The block-level log-sum-exp objective had an exact top-k solution. This was a useful negative result: GA was unnecessary for that objective.

### 4. Proxy block selection avoided full token-pair scoring

The key practical step was to score pooled query/key block summaries rather than materializing the full token-token score matrix before selection.

In an attention-only benchmark, Proxy Block-CAGE crossed over around 6K context and became faster at longer contexts.

At 16,384 tokens:

```text
dense attention:       ~14.98 ms
Proxy Block-CAGE 64:    ~3.97 ms
```

### 5. The speedup survived inside a Transformer block

A more realistic block benchmark included embedding, positional embedding, LayerNorm, QKV projection, attention, output projection, MLP, and residuals.

At 16,384 tokens:

```text
dense block:           ~18.78 ms
Proxy Block-CAGE 64:    ~5.74 ms
hidden-state MSE:       ~0.001
```

### 6. The speedup survived a trained real-text sanity check

The final sanity check trained a tiny 2-layer character-level Transformer on a custom corpus, copied the same trained weights into Proxy Block-CAGE variants, and measured generation-style prefill.

The best 16,384-token real-text result was:

```text
dense prefill:         ~24.30 ms
Proxy Block-CAGE 64:    ~9.32 ms
speedup:                2.61x
density:                0.39%
top-1 agreement:        100%
top-5 containment:      100%
```

Again: this is not production LLM evidence. It is a tiny trained model sanity check.

---

## Genetic programming for local attention kernels

After Proxy Block-CAGE started working, I tried another out-of-the-box idea: use genetic programming to search for cheaper local attention kernels.

Standard softmax uses:

$$
g(x)=\exp(x)
$$

where:

$$
x_{ij}=s_{ij}-\max_j s_{ij}
$$

The GP search evolved scalar functions \(g(x)\) using primitives such as:

```text
+ - * / max min avg exp log cos sin x^2 x^3 1/x 1/x^2
```

The normalized GP attention was:

$$
a_{ij}^{GP}
=
\frac{g(x_{ij})}{\sum_{\ell\in J_i}g(x_{i\ell})}
$$

The search rediscovered \(\exp(x)\) as the accuracy optimum, which is expected because the target was softmax attention. More interestingly, it found simple non-exp approximations such as:

$$
g(x)=\frac{1}{(1-x)^2}
$$

$$
g(x)=\max(1+0.25x,0.125)^2
$$

and a polynomial/max candidate:

$$
g(x)=\max\left(1.5+x,\max(1+0.25x,0.125)^2\right)
$$

These are mathematically interesting because they avoid expensive transcendental functions. But when inserted into Proxy Block-CAGE in unfused PyTorch, they were **not faster** than local softmax. At 16,384 tokens, Proxy Block-CAGE with local softmax remained the best version:

```text
dense:                  ~23.55 ms
CAGE + local softmax:     ~9.91 ms
CAGE + rational kernel:  ~11.46 ms
CAGE + clipquad kernel:  ~11.34 ms
CAGE + polymax kernel:   ~11.89 ms
```

I am treating the GP-kernel work as an early negative/possible future result. It may only become useful in a fused Triton/CUDA kernel where the operation count matters more directly.

---

## How to run the final script

This article is paired with a single Python file:

```text
proxy_block_cage_final.py
```

No separate vocabulary file or token file is needed.

The script builds a **character-level vocabulary** directly from the text corpus. It also includes an embedded fallback corpus, so it can run even if no text file is provided.

### Requirements

```text
Python 3.10+
PyTorch with CUDA recommended
```

The experiments above used:

```text
Python: 3.14.6
PyTorch: 2.12.1+cu126
GPU: NVIDIA GeForce RTX 2080 with Max-Q Design
CUDA from PyTorch: 12.6
dtype: fp16
```

### Optional: extract the embedded corpus

```bash
python proxy_block_cage_final.py --write-embedded-corpus my_corpus.txt
```

This writes the embedded corpus to `my_corpus.txt` and exits.

### Quick smoke test

```bash
python proxy_block_cage_final.py \
  --device cuda \
  --dtype fp16 \
  --seq-len 512 \
  --batch 4 \
  --train-steps 50 \
  --eval-iters 5 \
  --model-dim 128 \
  --heads 4 \
  --layers 1 \
  --block-cage-sizes 2 \
  --kernels softmax \
  --output-dir quick_cage_run
```

### Reproduce the 16K-style run

```bash
python proxy_block_cage_final.py \
  --device cuda \
  --dtype fp16 \
  --seq-len 16384 \
  --batch 1 \
  --train-steps 200 \
  --eval-iters 8 \
  --model-dim 256 \
  --heads 4 \
  --layers 2 \
  --block-cage-sizes 2 \
  --kernels softmax \
  --warmups 2 \
  --repeats 20 \
  --output-dir cage_text_quality_16384_repeat
```

On Windows Command Prompt, use `^` instead of `\` for line continuation:

```bat
python proxy_block_cage_final.py ^
  --device cuda ^
  --dtype fp16 ^
  --seq-len 16384 ^
  --batch 1 ^
  --train-steps 200 ^
  --eval-iters 8 ^
  --model-dim 256 ^
  --heads 4 ^
  --layers 2 ^
  --block-cage-sizes 2 ^
  --kernels softmax ^
  --warmups 2 ^
  --repeats 20 ^
  --output-dir cage_text_quality_16384_repeat
```

### Use your own text file

```bash
python proxy_block_cage_final.py \
  --device cuda \
  --dtype fp16 \
  --text-file my_corpus.txt \
  --seq-len 8192 \
  --batch 1 \
  --train-steps 300 \
  --eval-iters 20 \
  --model-dim 256 \
  --heads 4 \
  --layers 2 \
  --block-cage-sizes 2 \
  --kernels softmax \
  --warmups 2 \
  --repeats 20 \
  --output-dir cage_text_quality_8192
```

The script writes:

```text
cage_gp_kernel_quality.csv
cage_gp_kernel_quality.json
```

The JSON contains the environment, config, training history, and result rows.

---

## What the code compares

The final script trains a tiny dense character-level Transformer, then evaluates:

```text
dense
proxy_topk_softmax
proxy_topk_rational
proxy_topk_clipquad
proxy_topk_polymax
```

The main method is:

```text
proxy_topk_softmax
```

That means Proxy Block-CAGE uses top-k block routing and exact local softmax inside the selected blocks.

The GP kernels are included as experimental variants:

```text
rational: g(x) = 1 / (1 - x)^2
clipquad: g(x) = max(1 + 0.25x, 0.125)^2
polymax:  g(x) = max(1.5 + x, max(1 + 0.25x, 0.125)^2)
```

They are not the current speed winner.

---

## What is novel here?

I do not want to overclaim novelty. Sparse attention is a large area. Related ideas include block-sparse attention, routing attention, local/global attention, token pruning, approximate attention, hierarchical sparse attention, and newer long-context systems.

The narrow contribution here is:

> An independently developed, ChatGPT-assisted research prototype that starts from an evolutionary attention-mask formulation, discovers that the simple GA objective collapses to top-k, and converges to a training-free pooled block-routing sparse attention method with reproducible consumer-GPU long-context experiments.

I am not claiming to have invented sparse attention. I am claiming that this specific research path, implementation, and set of small but reproducible experiments are worth discussing.

---

## Limitations

This project has many limitations.

First, the strongest trained-model experiments use a tiny character-level Transformer, not a pretrained LLM.

Second, the eval sample counts are small. The top-1/top-5 agreement numbers should be interpreted as sanity-check signals, not definitive language-model quality results.

Third, the implementation is a PyTorch prototype, not a fused CUDA/Triton kernel.

Fourth, memory usage is worse than dense attention in the current implementation. At 16K, the sparse version was faster, but peak memory was higher. This is likely due to gather/local-attention intermediate tensors.

Fifth, training/backward is not solved. In backward tests, dense attention was faster and used much less memory.

Sixth, the current routing step still scores all block pairs. With fixed block size, the algorithm is not truly linear-time attention. It is reduced quadratic block routing plus sparse local attention.

Seventh, the GP-discovered local kernels are not a current speed win in PyTorch. They are included as a negative/early result and possible future direction.

---

## What I would like help with

I am sharing this because I want honest feedback and collaboration.

I am especially interested in hearing from people working on:

```text
sparse attention
long-context inference
LLM systems
CUDA / Triton kernels
routing attention
approximate attention
GPU benchmarking
Transformer architecture research
AI-assisted scientific workflows
```

The next serious steps would be:

1. Compare against stronger sparse-attention baselines.
2. Test on a real pretrained model rather than a tiny character model.
3. Add multiple seeds and stronger quality metrics.
4. Implement the sparse local attention path as a fused Triton/CUDA kernel.
5. Fix training/backward memory.
6. Explore learned or cached routing.
7. Revisit genetic algorithms only for richer non-top-k objectives, such as diversity, layout, locality, cache reuse, or latency-aware routing.

I am a GMU PhD student trying to pivot from applied AIML engineering into core AIML research. I would welcome academic collaboration, prior-art feedback, and serious technical criticism to refine and improve this work.

You can reach me at:

```text
iaddou@gmu.edu
cto@ezducate.ai
```

