# @Time: 2024/9/30 19:14
# @Author: xy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from configs.model_config import *

seed = 42  # 可以选择任意整数作为种子
torch.manual_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(BGE_M3_PATH)
model = AutoModel.from_pretrained(BGE_M3_PATH)
model.eval()


class MyAttention(torch.nn.Module):
    def __init__(self, num_head, dim_in, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.num_head = num_head
        self.wq = nn.Linear(in_features=dim_in, out_features=num_head * dim_out)
        self.wk = nn.Linear(in_features=dim_in, out_features=num_head * dim_out)
        self.wv = nn.Linear(in_features=dim_in, out_features=num_head * dim_out)
        self.dense = nn.Linear(num_head * dim_out, dim_out)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.wq, self.wk, self.wv, self.dense]:
            nn.init.xavier_uniform_(layer.weight)  # 使用 Xavier 初始化
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, dim]
        :return: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim_in = inputs.shape
        Q = self.wq(inputs)
        K = self.wk(inputs)
        V = self.wv(inputs)

        # 把不同头的Q,K,V拆分一下
        Q = Q.view(batch_size, seq_len, self.num_head, self.dim_out).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_head, self.dim_out).transpose(1, 2).transpose(2, 3)
        V = V.view(batch_size, seq_len, self.num_head, self.dim_out).transpose(1, 2)

        # 计算P,O, 注意einsum的结果部分(->后面那部分)不可以出现重复的字母
        P = torch.einsum('bnsd, bndS->bnsS', Q, K) / (self.dim_out ** 0.5)  # [b,n,s,s]
        P = nn.functional.softmax(P, dim=-1)  # [b,n,s,s]
        O = torch.einsum('bnss,bnsd->bnsd', P, V)  # [b,n,s,d]
        # concat
        O = O.transpose(1, 2).contiguous()
        O = O.view(batch_size, seq_len, self.num_head * self.dim_out)
        O = self.dense(O)
        return O


if __name__ == "__main__":
    query = ["中华人民共和国", "成立了"]
    encoded_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    last_hidden_state = model_output[0]
    # print(last_hidden_state.shape)  # [2,4,1024]
    # the input of transformer block
    inputs = last_hidden_state
    # feed into transformer block
    O = MyAttention(num_head=2, dim_in=1024, dim_out=256)(inputs)
    print(O[0][:, 1])
