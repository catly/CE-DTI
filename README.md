## CE-DTI:
Causal Enhanced Drug-Target Interaction Prediction based on Graph Generation and Multi-Source Information Fusion
### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./train.py`: predict drug-target interactions. 

```Python
python train.py --learning_rate=0.001 --dataname=zheng --dropout=0.2 inp_size=128
```
We can download the source code and data using Git.

```
git clone https://github.com/catly/CE-DTI.git
```




![1716038414569](https://github.com/catly/CE-DTI/assets/9825370/3e8c74e4-75e9-4773-9fbb-aabc262c23a4)






### envs

|Name                    |Version               | 
|-------|-------|
|python          |          3.8.0      |        
|dgl              |         0.8.0       |      
|networkx        |         3.1       |       
|numpy            |         1.24.3    |      
|pandas          |          2.0.3     |      
|scikit-learn    |          1.3.0      |         
|scipy          |           1.10.1     |           
|torch        |             1.9.0+cu111  |          
|torchaudio   |             0.9.0         |          
|torchvision    |           0.10.0+cu111 |            



### Code and data
The file structure is as follows: 
```
CE-DTI

└───data

│   └─── heter

│   └─── zheng

└───trainmodel.py

└───cmodel.py

└───utils.py

└───getEmbFromPubMedBERT.py


```


#### `data/` directory
##### [heter](https://github.com/luoyunan/DTINet)
The heter dataset mentioned in the text can be found in [heter](https://github.com/luoyunan/DTINet).
- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of disease names
- `se.txt`: list of side effect names
- `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
- `mat_drug_se.txt` 		: Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_protein_drug.txt` 	: Protein-Drug interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt` 		: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` 	: Drug-Disease association matrix
- `Similarity_Matrix_Drugs.txt` 	: Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` 	: Protein similarity scores based on primary sequences of proteins
- `drug_description.txt` : Drug text description information
- `drug_description_emb.pkl` : Embeddings obtained from drugs through the pre-trained model.
  

##### [Zheng](https://opus.lib.uts.edu.au/bitstream/10453/133212/4/2844947A-867E-4FFE-B718-ED9D728E0F76%20am.pdf)
The Zheng dataset mentioned in the text can be found in [Zheng](https://opus.lib.uts.edu.au/bitstream/10453/133212/4/2844947A-867E-4FFE-B718-ED9D728E0F76%20am.pdf).
- `drug_dic`: a complete ID mapping between drug names and DrugBank ID
- `target_dic`: a complete ID mapping between target names and UniProt ID
- `mat_drug_sideeffects.txt` 		: Drug-SideEffect association matrix
- `mat_drug_target.txt` 	: Drug-Target interaction matrix
- `mat_target_GO.txt` : Target-GO association matrix
- `mat_drug_chemical_sim` 	: Drug similarity scores based on chemical structures of drugs
- `mat_target_GO_sim.txt` 	: Target similarity scores based on Gene Ontology 
- `drug_description.txt` : Drug text description information
- `drug_description_emb.pkl` : Embeddings obtained from drugs through the pre-trained model.

#### `train.py` 
The train.py file contains the code for model training, evaluation, and prediction.

#### `cmodel.py` 
The model code for graph generation, multi-source information fusion, and drug-target relationship prediction is stored.

#### `utils.py` 
The utils.py file contains utility classes, data reading functions, graph structure construction methods, and evaluation approaches.

#### `getEmbFromPubMedBERT.py` 
The getEmbFromPubMedBERT.py file contains text processing and code for converting to embeddings using PubMedBERT.

### Pretrain-Model
The PubMedBERT model mentioned in the paper can be found in [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).








