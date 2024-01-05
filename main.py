import os
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import uuid
import chromadb
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

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


# Inizializza il tokenizer e il modello BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Funzione per calcolare gli embeddings BERT
def calcola_embeddings(testo):
    inputs = tokenizer(testo, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()


# Se i file "embeddings_testi.npy" e "embeddings_originali.npy" non sono stati ancora generati per salvare in locale
# il prodotto finale degli embeddings creati a partire dal file di testo, procede a creare tali file
if not (os.path.exists("embeddings_testi.npy") and os.path.exists("embeddings_originali.npy")):

    # Calcola gli embeddings per ogni testo
    embeddings = [calcola_embeddings(testo) for testo in text_documents]

    # Concatena gli embeddings in un unico tensore
    embeddings_tensor = torch.cat(embeddings, dim=0)

    # Conversione in numpy per PCA
    embeddings_numpy = embeddings_tensor.numpy()

    # Salva gli embeddings originali di 768 caratteristiche
    np.save("embeddings_originali.npy", embeddings_numpy)

    # Applica PCA per ridurre a 3 componenti
    pca = PCA(n_components=3)
    embeddings_3 = pca.fit_transform(embeddings_numpy)

    # Verifica delle dimensioni degli embeddings ridotti
    print(embeddings_3.shape)  # Dovrebbe essere (1000, 3)

    np.save("embeddings_testi.npy", embeddings_3)

# Se, invece, il file "embeddings_testi.npy" è già stato creato, evita di effettuare tutto il procedimento di creazione,
# decisamente esoso in termini di tempo e risorse, richiamando il file già esistente
else:

    # Carica gli embeddings originali di 768 caratteristiche per addestrare PCA
    embeddings_numpy = np.load("embeddings_originali.npy")

    # Carica gli embeddings ridotti
    embeddings_3 = np.load("embeddings_testi.npy")
    # print(embeddings_3.shape)  # Dovrebbe essere (1000, 3)

    # Addestra PCA sugli embeddings originali
    pca = PCA(n_components=3)
    pca.fit(embeddings_numpy)


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


matplotlib.use('TkAgg')  # Imposta TkAgg come backend

plt.ion()

# Estrae le componenti separate
x = embeddings_3[:, 0]  # Prima componente
y = embeddings_3[:, 1]  # Seconda componente
z = embeddings_3[:, 2]  # Terza componente

# Crea una figura per il plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Aggiunge i dati al grafico
ax.scatter(x, y, z)

# Imposta le etichette degli assi
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

#  Modifica il titolo del grafico
fig.canvas.manager.set_window_title('Visualizzazione tridimensionale degli embeddings')

# Mostra il grafico
plt.show()


# Funzione per caricare un nuovo file di testo
def carica_file_testo():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tk
    new_file_path = filedialog.askopenfilename()  # Apre il dialogo per la selezione del file
    if new_file_path:
        with open(new_file_path, 'r', encoding="utf8") as new_file:
            return new_file.read()
    return None


# Carica il nuovo testo e verifica che non sia vuoto
nuovo_testo = carica_file_testo()
if nuovo_testo:
    # Calcola l'embedding per il nuovo testo
    nuovo_embedding = calcola_embeddings(nuovo_testo)
    nuovo_embedding_np = nuovo_embedding.numpy()[0]  # Conversione in numpy
    nuovo_embedding_3 = pca.transform([nuovo_embedding_np])[0]  # Applica PCA

    # Converte il nuovo embedding in una lista
    nuovo_embedding_lista = nuovo_embedding_3.tolist()

    # Aggiunge il nuovo embedding alla collezione
    nuovo_id_uuid = str(uuid.uuid4())
    collection.add(
        embeddings=[nuovo_embedding_lista],
        documents=[nuovo_testo],
        ids=[nuovo_id_uuid]
    )

    # Aggiunge il nuovo embedding al grafico
    ax.scatter(nuovo_embedding_3[0], nuovo_embedding_3[1], nuovo_embedding_3[2], color='red')
    plt.draw()  # Aggiorna il grafico con il nuovo punto
    plt.pause(0.1)  # Dà tempo a Matplotlib di processare gli eventi della GUI

    print(f"Nuovo embedding aggiunto con ID: {nuovo_id_uuid}")
else:
    print("Nessun file caricato.")

# Mantiene aperta la finestra del grafico
plt.show(block=True)
