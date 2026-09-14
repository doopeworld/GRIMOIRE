# Extracted from huggingface/transformers modeling_rope_utils.py, main
# branch, 2026-09-14.  This is the ONLY rope type gemma-4 full-attention
# layers use, and it is NOT the same as GRIMOIRE's partial_rope:
#
#   proportional:  inv_freq = base ** -(2i / HEAD_DIM)  for i < rope_angles,
#                  0 for the rest, and the cos/sin table stays head_dim/2
#                  wide -- so dim i pairs with dim i + head_dim/2.
#   GRIMOIRE:      inv_freq = theta ** -(2j / ROT)      with ROT = head_dim
#                  * partial_factor, and dim j pairs with dim j + ROT/2.
#
# Different frequencies AND a different dimension pairing.  Substituting
# one for the other loads cleanly and produces the wrong model's output.
def _compute_proportional_rope_parameters(
    config: Optional["PreTrainedConfig"] = None,
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    head_dim_key: str = "head_dim",
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with proportional RoPE.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`, *optional*): The base wavelength from which the inverse frequencies will be derived. Defaults to `config.default_theta` if omitted.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*, defaults to 1.0): The proportion of the embedding dimension
                to apply rotary positional encoding, e.g., [0.0, 0.25, 0.5, 0.75, 1.0]. Unlike other RoPE functions
                that use this parameter, proportional RoPE will always return an encoding that is the size of
                `head_dim`.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Sloppy workaround for per-layer-config in gemma4 family which raises error for other models (DS4)
    try:
        config = config.per_layer_config[layer_type] if layer_type is not None else config
    except ValueError:
        pass

    # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
    config.standardize_rope_params()
    rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters

    head_dim = getattr(config, head_dim_key, None) or config.hidden_size // config.num_attention_heads
    base = rope_parameters_dict["rope_theta"]
    factor = rope_parameters_dict.get("factor", 1.0)
    rope_proportion = rope_parameters_dict.get("partial_rotary_factor", 1.0)

    attention_factor = 1.0  # Unused in this type of RoPE

    rope_angles = int(rope_proportion * head_dim // 2)

    inv_freq_rotated = 1.0 / (
        base
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
    )

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = torch.cat(
            (
                inv_freq_rotated,
                torch.zeros(nope_angles, dtype=torch.float32, device=device),
            ),
            dim=0,
        )
    else:
        inv_freq = inv_freq_rotated

    inv_freq /= factor
    return inv_freq, attention_factor


