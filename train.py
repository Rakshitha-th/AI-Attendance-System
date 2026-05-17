import os
import cv2
import numpy as np
import joblib

from collections import defaultdict
from insightface.app import FaceAnalysis
from sklearn.preprocessing import Normalizer

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)


dataset_path = "Dataset"

# store embeddings person-wise
person_embeddings = defaultdict(list)
people = os.listdir(dataset_path)

for person in people:
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        try:

            faces = app.get(img)

            if len(faces) == 0:
                continue

            embedding = faces[0].embedding

            person_embeddings[person].append(embedding)

            print(f"Processed: {person} - {image_name}")

        except:
            continue
known_embeddings = []
known_names = []

for person, embeds in person_embeddings.items():

    mean_embedding = np.mean(embeds, axis=0)
    known_embeddings.append(mean_embedding)
    known_names.append(person)


known_embeddings = np.array(known_embeddings)
normalizer = Normalizer(norm='l2')
known_embeddings = normalizer.transform(known_embeddings)

os.makedirs("models", exist_ok=True)

joblib.dump(known_embeddings, "models/embeddings.pkl")
joblib.dump(known_names, "models/names.pkl")
joblib.dump(normalizer, "models/normalizer.pkl")

print("\nEmbeddings Saved Successfully")
print("Total Persons:", len(known_names))