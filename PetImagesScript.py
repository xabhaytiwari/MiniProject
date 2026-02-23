import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SubModularFunction(ABC):
    @abstractmethod
    def evaluate(self, selected_indices: List[int]) -> float:
        pass

    @abstractmethod
    def marginal_gain(self, selected_indices: List[int], candidate_index: int) -> float:
        pass

class LabelSampleCosine(SubModularFunction):
    def __init__(self, X_pool: np.ndarray):
        super().__init__()

        self.similarity_matrix = cosine_similarity(X_pool)
        self.similarity_matrix = np.clip(self.similarity_matrix, 0.0, 1.0)
    
    def evaluate(self, selected_indices: List[int]) -> float:
        if not selected_indices:
            return 0.0
        sub_matrix = self.similarity_matrix[:, selected_indices]
        max_similarities = np.max(sub_matrix, axis=1)
        return np.sum(max_similarities)
    
    def marginal_gain(self, selected_indices, candidate_index):
        current_score = self.evaluate(selected_indices)
        new_indices = selected_indices + [candidate_index]
        new_score = self.evaluate(new_indices)
        return new_score - current_score

class Greedyptimizer:
    def __init__(self, objective: SubModularFunction):
        self.objective = objective
    
    def select(self, total_size: int, budget: int) -> List[int]:
        selected_indices = []
        remaining_incdices = list(range(total_size))

        for _ in range(budget):
            best_gain = float('-inf')
            best_candidate = -1

            for candidate in remaining_incdices:
                gain = self.objective.marginal_gain(selected_indices, candidate)

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
                remaining_incdices.remove(best_candidate)
        
        return selected_indices

extensions = ["jpg", "png", "jpeg"]
image_paths = []

for ext in extensions:
    image_paths.extend(
        glob.glob(f"PetImages/**/*.{ext}", recursive=True)
    )


batch_size = 32
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

all_embeddings = []

with torch.no_grad():
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(path).convert('RGB') for path in batch_paths]

        embeddings = model.encode(images)
        all_embeddings.append(embeddings)

all_pro_embeddings = np.vstack(all_embeddings)

objective = LabelSampleCosine(all_pro_embeddings)
optimizer = Greedyptimizer(objective)

print("Calculating the most representative images. Please wait.")

selected_indices = optimizer.select(total_size=len(all_pro_embeddings), budget=100)

user_labels = []

print("Beginning interactive labeling session.")

for i, index in enumerate(selected_indices):
        current_path = image_paths[index]
        
        img = Image.open(current_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}")
        plt.show(block=False)
        plt.pause(0.1)

        while True:
            user_input = input("Enter label type cat or dog: ").strip().lower()
            if user_input in ['cat', 'dog']:
                label = 0 if user_input == 'cat' else 1
                user_labels.append(label)
                break
            else:
                print("Invalid input. Type cat or dog.")
        
        plt.close()

print("Training the classifier.")

training_features = all_pro_embeddings[selected_indices]
training_labels =  np.array(user_labels)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(training_features, training_labels)

true_labels = []

for path in image_paths:
    if 'dog' in path.lower():
            true_labels.append(1)
    elif 'cat' in path.lower():
            true_labels.append(0)
    else:
            raise ValueError("Path must contain either cat or dog directory names.")

true_labels = np.array(true_labels)

predictions = classifier.predict(all_pro_embeddings)
accuracy = accuracy_score(true_labels, predictions)

print(f"Final Model Accuracy on all {len(all_pro_embeddings)} images: {accuracy * 100:.2f} percent")