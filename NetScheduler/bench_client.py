# bench_client.py
import asyncio, time, json, random
import aiohttp
from statistics import mean, median

SERVER="http://127.0.0.1:8000"   # 按需改
STREAM_EP=f"{SERVER}/generate_stream"
BATCH_EP =f"{SERVER}/generate"

def make_prompt(n_tokens:int):
    return " ".join(["hello"]*n_tokens)

async def fire_one(session, prompt, max_new_tokens=128):
    t0=time.perf_counter()
    # 试流式
    try:
        async with session.post(STREAM_EP, json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "stream": True
        }) as resp:
            ttft=None
            out_tokens=0
            async for line in resp.content:
                if not line:
                    continue
                if ttft is None:
                    ttft=time.perf_counter()-t0
                # 这里简单按行计 token；若服务端是 JSON chunk，解析后加一
                out_tokens+=1
            tdone=time.perf_counter()
            return {"ttft":ttft if ttft is not None else (tdone-t0),
                    "out_tokens":out_tokens, "lat":tdone-t0}
    except Exception:
        pass

    # 回落非流式
    async with session.post(BATCH_EP, json={
        "prompt": prompt,
        "max_new_tokens": max_new_tokens
    }) as resp:
        data=await resp.json()
        tdone=time.perf_counter()
        text=data.get("text","")
        out_tokens=len(text.split())
        return {"ttft":tdone-t0, "out_tokens":out_tokens, "lat":tdone-t0}

async def run_qps(session, qps:int, dur_s:int, prompt_len:int, max_new_tokens:int):
    end=time.perf_counter()+dur_s
    inter=1.0/max(qps,1)
    tasks=[]
    while time.perf_counter()<end:
        tasks.append(asyncio.create_task(
            fire_one(session, make_prompt(prompt_len), max_new_tokens)
        ))
        await asyncio.sleep(inter)
    res=await asyncio.gather(*tasks, return_exceptions=True)
    res=[r for r in res if isinstance(r,dict)]
    if not res: return {}
    ttfts=[r["ttft"] for r in res if r["ttft"] is not None]
    tpops=[(r["out_tokens"]/max(r["lat"],1e-6)) for r in res if r["out_tokens"]>0]
    return {
        "qps": qps,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "reqs": len(res),
        "avg_ttft": mean(ttfts) if ttfts else None,
        "p50_ttft": median(ttfts) if ttfts else None,
        "avg_tpop_tok_s": mean(tpops) if tpops else None,
    }

async def main():
    async with aiohttp.ClientSession() as sess:
        print("=== W1: same length, varying QPS ===")
        for q in [1,2,4,8,16,32]:
            stats=await run_qps(sess, qps=q, dur_s=30, prompt_len=512, max_new_tokens=128)
            print(json.dumps(stats))

        print("=== W2: fixed QPS=8, varying prompt len ===")
        for L in [64,256,512,1024,2048]:
            stats=await run_qps(sess, qps=8, dur_s=30, prompt_len=L, max_new_tokens=128)
            print(json.dumps(stats))

        print("=== W3: (TTFT/TPOP reported above) ===")

if __name__=="__main__":
    asyncio.run(main())
