## Resource
|Resource Size|Latency|
|--|--|
|2|522.5829703807831|
|4|525.9594669342041|
|6|526.1527788639069|
|8|529.1130328178406|
|||

## rank-size

|Rank Size|Latency|
|--|--|
|8|183.57363319396973|
|16|207.9325888156891|
|32|219.98093438148499|
|64|224.96146416664124|
|128|251.7184820175171|
|256|277.1999180316925|


## Scale-Up vs Scale-Out



### LoRA adapter size
- consider: q_proj, k_proj, v_proj, o_proj
- Llama2-7b as base model: 12.7GB
- per layer: 4 * 2 * rank * hidden_size (4096)

|rank|adapter size|
|--|--|
|8|32MB|
|16|64MB|
|32|128MB|
|64|258MB|
|128|512MB|

- 为了更好比较，以KV cache size per layer作为对照组: 2 * num_tokens * hidden_size
    - size per LoRA adapter = 4 * rank * KV-Cache-Size(per token)