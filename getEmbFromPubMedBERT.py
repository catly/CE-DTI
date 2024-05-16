from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModel
import pandas as pd
set_seed(42)
def getEmbFromPubMedBERT():
    tokenizer = AutoTokenizer.from_pretrained("../pubMedBERT")
    model = AutoModel.from_pretrained("../pubMedBERT")
    text = open("./drug_description.txt","r").readlines()
    text = [i.strip() for i in text]
    inputs = tokenizer(text,max_length = 96,padding=True,add_special_tokens=True,return_tensors = "pt").input_ids
    res =model(inputs)
    pd.to_pickle(res.pooler_output.detach(),"./biogptemb.pkl")
    return res

getEmbFromPubMedBERT()
res = pd.read_pickle("./description.pkl")



