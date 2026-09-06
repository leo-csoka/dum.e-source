class Config:
    def __init__(
        self,
        token_vector_len=768,
        vocab_size=100277,
        context_length=512,
        head_dim=64,
        num_blocks=12,
        ffn_dim=2048,
        hidden_upscale=None,
    ):
        # model hyperparameters
        self.token_vector_len = token_vector_len
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.head_dim = head_dim
        self.num_heads = self.token_vector_len // self.head_dim
        # Accept the old config field when loading existing checkpoints.
        if hidden_upscale is not None:
            ffn_dim = round(hidden_upscale * self.token_vector_len)
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks

        if self.token_vector_len % self.head_dim != 0:
            raise ValueError("token_vector_len must be divisible by head_dim")