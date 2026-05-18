try:
    from sglang.test.test_utils import is_in_ci
    import sglang as sgl

    if is_in_ci():
        import patch
    else:
        import nest_asyncio

        nest_asyncio.apply()
except:
    pass

def get_sglang_llm(args):
    llm = sgl.Engine(
        model_path=args.model_name_or_path,
        tp_size=args.tensor_parallel_size,
        dp_size=args.data_parallel_size,
        trust_remote_code=True,
        max_prefill_tokens=32768,
        random_seed=args.seed,
        chunked_prefill_size=-1,
        schedule_policy="lpm",
        show_time_cost=True,
        num_continuous_decode_steps=1,
        schedule_conservativeness=1,
        attention_backend="flashinfer",
        context_length=32768,
        mem_fraction_static=0.8,
    )
    return llm
