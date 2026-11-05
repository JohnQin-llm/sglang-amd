"""Apply the K2 gfx1250 hooks to a COPY of the served rocm_muse_spark/model.py (never the served tree).

    python3 apply_hooks.py <model.py copy>

Hooks (all behind SGLANG_MUSE_MONO_GFX1250=1; without it the file behaves exactly as served):
  1. module flag + runner handle;
  2. MuseSpark1xAttention.forward(..., core_only=False): return the o_proj input when core_only;
  3. MuseSpark1xDecoderLayer.forward: K2 (o_proj .. MoE branch) for the layers / steps the runner takes, then the
     native FFN boundary;
  4. MuseSpark1xModel.forward: advance the K2 epoch once a step K2 runs in;
  5. post_load_weights: build the runner (collective over the TP group) and the per-layer K2 operands.
"""

import sys


def sub(src, old, new, count=1):
    n = src.count(old)
    if n != count:
        raise SystemExit(f"hook anchor found {n} times (want {count}):\n{old}")
    return src.replace(old, new)


def main(path):
    src = open(path).read()

    src = sub(
        src,
        "from sglang.srt.utils import add_prefix\n",
        "from sglang.srt.utils import add_prefix\n"
        "\n"
        "# K2 gfx1250 persistent per-layer kernel (o_proj .. MoE branch), TP4 / EP1 expert copy.\n"
        "_K2G = os.environ.get(\"SGLANG_MUSE_MONO_GFX1250\", \"0\") == \"1\"\n"
        "_K2G_RUNNER = None  # built in post_load_weights\n",
    )

    src = sub(
        src,
        "        forward_batch: ForwardBatch,\n"
        "        reduce: bool = True,\n"
        "    ) -> torch.Tensor:\n",
        "        forward_batch: ForwardBatch,\n"
        "        reduce: bool = True,\n"
        "        core_only: bool = False,\n"
        "    ) -> torch.Tensor:\n",
    )
    src = sub(
        src,
        "        output = self._fused_o_norm_gate.forward(output, o_gate)\n"
        "        output, _ = self.o_proj(output, skip_all_reduce=not reduce)\n",
        "        output = self._fused_o_norm_gate.forward(output, o_gate)\n"
        "        if core_only:\n"
        "            return output\n"
        "        output, _ = self.o_proj(output, skip_all_reduce=not reduce)\n",
    )

    src = sub(
        src,
        "        attn_input = (\n"
        "            self.self_attn_input_norm(stream) if attn_input is None else attn_input\n"
        "        )\n",
        "        attn_input = (\n"
        "            self.self_attn_input_norm(stream) if attn_input is None else attn_input\n"
        "        )\n"
        "        k2r = _K2G_RUNNER\n"
        "        if (\n"
        "            k2r is not None\n"
        "            and sp is None\n"
        "            and ffn_boundary is not None\n"
        "            and k2r.takes(self.layer_idx, stream.shape[0])\n"
        "        ):\n"
        "            core = self.self_attn(positions, attn_input, forward_batch, core_only=True)\n"
        "            stream, branch = k2r.run(self.layer_idx, core, stream)\n"
        "            next_attn_input, stream, _ = ffn_boundary.forward(\n"
        "                stream,\n"
        "                branch,\n"
        "                branch_dtype=torch.bfloat16,\n"
        "                pre_norm_dtype=pre_norm_dtype,\n"
        "            )\n"
        "            return stream, next_attn_input\n",
    )

    src = sub(
        src,
        "            attn_input = self.layers[0].self_attn_input_norm(stream)\n"
        "            last = len(self.layers) - 1\n",
        "            attn_input = self.layers[0].self_attn_input_norm(stream)\n"
        "            last = len(self.layers) - 1\n"
        "            if (\n"
        "                _K2G_RUNNER is not None\n"
        "                and sp is None\n"
        "                and _K2G_RUNNER.takes_step(stream.shape[0])\n"
        "            ):\n"
        "                _K2G_RUNNER.begin_step()\n",
    )

    src = sub(
        src,
        "        boundaries = [layer._fused_attn_boundary for layer in self.model.layers]\n"
        "        boundaries += list(getattr(self.model, \"_ffn_boundaries\", None) or [])\n"
        "        for op in boundaries:\n"
        "            ensure = getattr(op, \"_ensure_cached_weights\", None)\n"
        "            if ensure is not None:\n"
        "                ensure(torch.bfloat16)\n",
        "        boundaries = [layer._fused_attn_boundary for layer in self.model.layers]\n"
        "        boundaries += list(getattr(self.model, \"_ffn_boundaries\", None) or [])\n"
        "        for op in boundaries:\n"
        "            ensure = getattr(op, \"_ensure_cached_weights\", None)\n"
        "            if ensure is not None:\n"
        "                ensure(torch.bfloat16)\n"
        "        if _K2G:\n"
        "            self._build_k2g_runner()\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def _build_k2g_runner(self) -> None:\n"
        "        global _K2G_RUNNER\n"
        "        from sglang.srt.rocm_muse_spark.mono_gfx1250.runner import K2Runner\n"
        "\n"
        "        tp = get_parallel().tp_group\n"
        "        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName\n"
        "        if tp.world_size != 4 or not arch.startswith(\"gfx1250\"):\n"
        "            raise ValueError(\"SGLANG_MUSE_MONO_GFX1250 needs gfx1250 at TP4\")\n"
        "        if self.model._ffn_boundaries is None:\n"
        "            raise ValueError(\"SGLANG_MUSE_MONO_GFX1250 needs MUSE_FUSE_FFN_BOUNDARY=1\")\n"
        "        dev = torch.device(\"cuda\", torch.cuda.current_device())\n"
        "        runner = K2Runner(tp.cpu_group, tp.rank_in_group, dev)\n"
        "        for i, layer in enumerate(self.model.layers):\n"
        "            if isinstance(layer.mlp, MuseSpark1xMoE):\n"
        "                runner.add_layer(i, layer)\n"
        "        _K2G_RUNNER = runner\n",
    )
    open(path, "w").write(src)


if __name__ == "__main__":
    main(sys.argv[1])
