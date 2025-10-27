import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class bulingual_Dataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lng,seq_len)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt =tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lng

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype= torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype= torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype= torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index)->any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens  = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens  = self.tokenizer_src.encode(tgt_text).ids

        enc_num_padding_token =  self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_token =  self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_token <0 or dec_num_padding_token<0 :
            raise ValueError('Sentence is too long')
        # add sos and eos to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_token, dtype=torch.int64)
            ]
        )
        # add eos to the decoder input 
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_token, dtype=torch.int64)
            ]
        )
        # add exos to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_token, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0)==self.seq_lem
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len
            "encoder_mask": (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int()& causal_mask(decoder_input.size(0)), #(1,1,seq_len) & (1,seq_len, seq_len)
            "label":label,
            "src_txt":src_text,
            "tgt_text":tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),daigonal=1).type(torch.int)
    return mask==0