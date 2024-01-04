import os
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import uuid
import chromadb


folder_path = 'dataset_testi'
text_documents = []

# List all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Read the content of the text document
        with open(file_path, 'r', encoding="utf8") as file:
            document_content = file.read()

            # Append the content to the vector
            text_documents.append(document_content)

if not os.path.exists("embeddings_testi.npy"):

    # Inizializza il tokenizer e il modello BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Funzione per calcolare gli embeddings BERT
    def calcola_embeddings(testo):
        inputs = tokenizer(testo, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach()

    # Calcola gli embeddings per ogni testo
    embeddings = [calcola_embeddings(testo) for testo in text_documents]

    # Concatena gli embeddings in un unico tensore
    embeddings_tensor = torch.cat(embeddings, dim=0)

    # Conversione in numpy per PCA
    embeddings_numpy = embeddings_tensor.numpy()

    # Applica PCA per ridurre a 512 componenti
    pca = PCA(n_components=512)
    embeddings_512 = pca.fit_transform(embeddings_numpy)

    # Verifica delle dimensioni degli embeddings ridotti
    print(embeddings_512.shape)  # Dovrebbe essere (1000, 512)

    np.save("embeddings_testi.npy", embeddings_512)

else:

    embeddings_512 = np.load("embeddings_testi.npy")
    print(embeddings_512.shape)  # Dovrebbe essere (1000, 512)


# Assumendo che embeddings_512 e text_documents abbiano la stessa lunghezza
num_items = len(embeddings_512)
ids_uuid = [str(uuid.uuid4()) for _ in range(num_items)]


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    embeddings=embeddings_512,
    documents=text_documents,
    ids=ids_uuid
)