# InferScheduler

| System | Strategy | Throughput Impact | Latency Impact (TBT) | Notes |
| - - - - - - - - - -| - - - - - - - - - - -| - - - - - - - - - - | - - - - - - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -|
| **Sarathi-Serve** | Stall-free batching | High | Low | Utilizes chunked-prefills to allow high throughput with minimal latency impact. |
| **FasterTransformer** | Decode prioritizing | Low | Low | Prioritizes decodes, optimizing for latency at the expense of throughput. |
| **vLLM** | Prefill prioritizing| High | High | Prioritizes prefills for throughput, leading to potential high TBT latency due to stalls. |
| **Orca** | Iteration-level batching with prefill prioritizing | Moderate | High | Introduces dynamic batching improving throughput but can increase TBT latency. |