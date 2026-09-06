class Config:
    def __init__(
        self,
        token_vector_len=1536,
        vocab_size=100277,
        context_length=1024,
        head_dim=64,
        num_blocks=12,
        hidden_upscale=4,
    ):
        # model hyperparameters
        self.token_vector_len = token_vector_len
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.hidden_upscale = hidden_upscale
        self.num_heads = self.token_vector_len // self.head_dim