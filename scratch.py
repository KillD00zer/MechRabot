from haystack.dataclasses import ChatMessage
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
msg = ChatMessage.from_user("Hello world")
res = model.encode([msg], return_dense=True, return_sparse=True)
print(type(res['lexical_weights']))
print(res['lexical_weights'])
