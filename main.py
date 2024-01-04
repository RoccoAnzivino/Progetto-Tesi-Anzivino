import os
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import uuid
import chromadb


folder_path = 'dataset_testi'
text_documents = []

# Scorre tutti i file della cartella "dataset_testi"
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # Legge il contenuto dei file di testo
        with open(file_path, 'r', encoding="utf8") as file:
            document_content = file.read()

            # Aggiunge il contenuto dei file di testo all'interno del vettore "text_documents"
            text_documents.append(document_content)

# Se il file "embeddings_testi.npy" non è stato ancora generato per salvare in locale il prodotto finale degli
# embeddings creati a partire dal file di testo, procede a creare tale file
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

    # Applica PCA per ridurre a 3 componenti
    pca = PCA(n_components=3)
    embeddings_3 = pca.fit_transform(embeddings_numpy)

    # Verifica delle dimensioni degli embeddings ridotti
    print(embeddings_3.shape)  # Dovrebbe essere (1000, 3)

    np.save("embeddings_testi.npy", embeddings_3)

# Se, invece, il file "embeddings_testi.npy" è già stato creato, evita di effettuare tutto il procedimento di creazione,
# decisamente esoso in termini di tempo e risorse, richiamando il file già esistente
else:

    embeddings_3 = np.load("embeddings_testi.npy")
    print(embeddings_3.shape)  # Dovrebbe essere (1000, 3)


# Assumendo che embeddings_512 e text_documents abbiano la stessa lunghezza, si procede a creare tanti id quanti sono
# gli embeddings e, quindi, i file di testo
num_items = len(embeddings_3)
ids_uuid = [str(uuid.uuid4()) for _ in range(num_items)]


chroma_client = chromadb.Client()

# Creazione della collezione tramite la libreria chroma
collection = chroma_client.create_collection(name="my_collection")

# Aggiunta degli embeddings nella collezione
collection.add(
    embeddings=embeddings_3,
    documents=text_documents,
    ids=ids_uuid
)
