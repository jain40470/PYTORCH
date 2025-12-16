import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self , d_model , vocab_size):  
        
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        
        # (batch , seq ) => (batch , seq , d_model )

        out = self.embedding(x)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self,seq_len,d_model,dropout):

        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros( seq_len , d_model )  # (seq_len , d_model)

        # even posn : sin( pos * 1000^(- 2i / dmodel) )
        # odd posn : cos( pos * 1000^(- 2i / dmodel) )

        pos = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1) # (seq_len , 1)
        
        # e ^ (2i * (-log(1000) / d_model) )   ==> 1000^(- 2i / dmodel)
        div = torch.exp( torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model) ) # (d_model/2)

        # for even
        pe[:, 0::2] = torch.sin(pos * div)  # (during multiplication it automatcaily broadcasst i.e (10,1) * (4) => (10,4))
        # for odd
        pe[:, 1::2] = torch.sin(pos * div)

        pe = pe.unsqueeze(0)  # ( 1 ,seq_len , d_model)

        self.register_buffer('pe', pe)  # (not as trainable but saved in model.dict)

        
    def forward(self , x) :
    
        out = x + (self.pe[ : , : x.shape[1] , : ]).requires_grad_(False)    # (here it auto broadcasts the pe to multiple batches)
        
        out = self.dropout(out)

        # we not want pe to be trained

        # x : (batch , seq , d_model )

        # pe[ : , : x.shape[1] , : ] : ( 1 , x.shape[1] , d_model )
        
        # so it auto broadcasts for other and like consider it for other seq in batch.

        return out


# dont consider dropout

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self , d_model , h , dropout):

        super().__init__()

        self.d_model = d_model 
        self.h = h

        assert d_model % h == 0  # make sure d_model is divisible by h

        self.d_k = d_model // h  # vector len seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv

        # we are defining w_q,q_k,w_v once , but later split it only for each head.

        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
    

    @staticmethod
    def attention(query, key, value, mask,dropout : nn.Dropout): # erased the dropout from here
       
        d_k = query.shape[-1] 

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # transpose(-2, -1) last two dim swap

        # query : (batch, h, seq_len, d_k)
        # key transpose : (batch, h, d_k , seq_len)   [Note if seq len differ tehn also work : Decoder , helpful when decoder worked in testing]

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # masked_fill_ : tensor inbuilt function

        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # (attention_scores @ value)

        # attention_scores : (batch, h, seq_len, seq_len)
        # value : (batch, h, seq_len ,  d_k,)    [since attention_scores last dim is from key and during tetsing both key and val come from encoder so no issues]

        # return attention scores which can be used for visualization

    
        return  (attention_scores @ value), attention_scores
    
    def forward(self , q , k , v , mask):

        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        query = query.transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # x , self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x , attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (contagious used to stopre in sequential) , (-1 is used to automatically infer)

        # Multiply by Wo

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)  #(with the help of w_o , out from diff heads may like connect with each other)


class FeedForwardBlock(nn.Module):

    def __init__(self,d_model, d_ff, dropout):

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)

        out = self.linear_1(x)
        out = self.dropout(torch.relu(out))  # relu has no training paramater so we can have it as a function
        out = self.linear_2(out)

        return out
    
class LayerNormalization(nn.Module):

    def __init__(self , features , eps:float = 10**-6):

        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter  (reqd grad = true se their gradients calculate but not ensure ki they will be updated)

    def forward(self , x):
        
        mean = x.mean(dim = -1 , keepdim = True)  # (batch , seq , d_model) => (batch , seq ,1) [so we are keeping dim = true for broadcasting]
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)

        # (x - mean) : (batch , seq , d_model) - (batch, seq_len, 1) => it automatically broadcasts

        # alpha : (d_model,) , (x - mean) : (batch , seq , d_model) 

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# features is like d_model (embedding dim)


class ResidualConnection(nn.Module):

    def __init__(self, features,dropout: float) -> None:
    
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        
        # x = self.norm(x + self.dropout(sublayer(x))) : post norm (in original)
        # x = x + Dropout(Sublayer(LayerNorm(x))) : pre norm 

        # pre norm : Improves gradient flow in deep stacks

        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self,features,attention_block ,feed_forward_block, dropout):

        super().__init__()

        self.attention_block = attention_block 
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
        self.feed_forward_block = feed_forward_block

    def forward(self , x , src_mask):

        x = self.residual_connections[0](x, lambda x_lambda : self.attention_block(x_lambda,x_lambda, x_lambda, src_mask))
        x = self.residual_connections[1](x, lambda x_lambda : self.feed_forward_block(x_lambda))
        # x = self.residual_connections[1](x, self.feed_forward_block) can use this also.

        return x
    
class Encoder(nn.Module):

    def __init__(self,features :int , layers: nn.ModuleList):

        super().__init__()
        
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features , attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        
        super().__init__()

        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connections[0](x, lambda x: self.attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)   

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:

        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
       
        super().__init__()
       
        self.encoder = encoder
        self.decoder = decoder
       
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
       
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
       
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):

        src = self.src_embed(src)
        src = self.src_pos(src)  # (batch, seq_len, d_model)
        
        return self.encoder(src, src_mask)  
    
    def decode(self, encoder_output , src_mask , tgt , tgt_mask):
       
      
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt) # (batch, seq_len, d_model)
       
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)   
    
    def project(self, x):
    
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        #     
        return self.projection_layer(x)
    
    def forward(self , src , tgt , src_mask , tgt_mask):

        src = self.src_embed(src)
        src = self.src_pos(src)  
        
        encoder_output =  self.encoder(src, src_mask) 

        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt) 
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)   

        out = self.projection_layer(decoder_output)

        return out

    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) :
   
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []

    for _ in range(N):
        
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model,encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(d_model,decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer