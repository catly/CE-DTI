## CE-DTI:
Causal Enhanced Drug-Target Interaction Prediction based on Graph Generation and Multi-Source Information Fusion
### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./train.py`: predict drug-target interactions. 

```Python
python train.py --learning_rate=0.001 --dataname=zheng --dropout=0.2 inp_size=128
```

### envs

|Name                    |Version               | 
|-------|-------|
|cudatoolkit       |        11.3.1      |      
|dgl              |         0.8.0       |      
|networkx        |         3.1       |       
|numpy            |         1.24.3    |      
|pandas          |          2.0.3     |      
|pillow          |          7.1.2     |      
|python          |          3.8.0      |        
|scikit-learn    |          1.3.0      |         
|scipy          |           1.10.1     |           
|tokenizers     |           0.13.3      |            
|torch        |             1.9.0+cu111  |          
|torchaudio   |             0.9.0         |          
|torchvision    |           0.10.0+cu111 |            





### Code and data

#### `data/` directory
##### [heter](https://github.com/luoyunan/DTINet)
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
  

##### Zheng

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
Code module for model training and prediction.

#### `cmodel.py` 
 Model code.


#### `utils.py` 
Model utility class.

#### `getEmbFromPubMedBERT.py` 
Text processing and embedding.










