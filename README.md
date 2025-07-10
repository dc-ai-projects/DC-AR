# [ICCV 2025] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer

\[[Paper](https://arxiv.org/abs/2507.04947)\] \[[Demo](https://dc-ar.hanlab.ai)\] \[[Project](https://hanlab.mit.edu/projects/dc-ar)\]

## News

- \[2025/6\] 🔥 DC-AR is accepted by ICCV 2025!

## Abstract

We introduce DC-AR, a novel masked autoregressive (AR) text-to-image generation framework that delivers superior image generation quality with exceptional computational efficiency. Due to the tokenizers' limitations, prior masked AR models have lagged behind diffusion models in terms of quality or efficiency. We overcome this limitation by introducing DC-HT - a deep compression hybrid tokenizer for AR models that achieves a 32× spatial compression ratio while maintaining high reconstruction fidelity and cross-resolution generalization ability. Building upon DC-HT, we extend MaskGIT and create a new hybrid masked autoregressive image generation framework that first produces the structural elements through discrete tokens and then applies refinements via residual tokens. DC-AR achieves state-of-the-art results with a gFID of **5.49** on MJHQ-30K and an overall score of **0.69** on GenEval, while offering **1.5-7.9×** higher throughput and **2.0-3.5×** lower latency compared to prior leading diffusion and autoregressive models.

## Release
- The code and pretrained models will be released after the legal review is completed.
