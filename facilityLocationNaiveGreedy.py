from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Any
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class SubModularFunction(ABC):
    """
    Abstract Base Class defining the structure of Submodular Functions.

    This class servers as a strict contract for all submodular functions implementations.
    """

    @abstractmethod
    def evaluate(self, selected_set_indices: List[int]) -> float:
        """
        Calculates the total score of a set.

        :param selectedSet: The Selected Set, for which we're calculating the score.
        :type selectedSet: List
        :return: Return the Score for the set.
        :rtype: float
        """
        pass

    @abstractmethod
    def marginal_gain(self, selected_set_indices: List[int], item_index: int) -> float:
        """
         The Selected Set Indices, for which we're calculating the score.

        :type selectedSet: List
        :param item: A particular Item Index
        :type item: Any
        :return: Returns the marginal gain for that item.
        :rtype: float
        """
        pass

class FacilityLocationObjective(SubModularFunction):
    """
    The objective is to measure the "representativeness" of an Subset against our Dataset.
    """
    def __init__(self, X_pool: np.ndarray ):
        super().__init__()
        self.metric = 'euclidean'
        distances = pairwise_distances(X_pool, metric=self.metric)
        sigma = np.mean(distances)
        self.similarity_matrix = np.exp(-distances / sigma)
    
    def evaluate(self, selected_indices: List[int]) -> float:
        num_of_samples = self.similarity_matrix.shape[0]
        if not selected_indices:
            current_score = 0.0
        else:
            max_similarity_current = np.max(self.similarity_matrix[:, selected_indices], axis=1)
            current_score = np.sum(max_similarity_current)
        
        return current_score
    
    def marginal_gain(self, selected_set_indices: List[int], item_index: int) -> float:
        new_indices = selected_set_indices + [item_index]
        return self.evaluate(new_indices) - self.evaluate(selected_set_indices) 
    
class GreedyOptimizer:
    def __init__(self, objective: SubModularFunction):
        self.objective = objective
        
    
    def select(self, total_size: int, budget: int) -> List[int]:
        selected_indices = []
        remaining_indices = list(range(total_size))

        for _ in range(budget):
            best_gain = -np.inf
            best_candidate = -1

            for candidate in remaining_indices:
                gain = self.objective.marginal_gain(selected_indices, candidate)
            
                if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)

        return selected_indices

if __name__ == "__main__":
    np.random.seed(42)
    data_pool = np.random.rand(20, 2)

    objective_function = FacilityLocationObjective(data_pool)
    optimizer = GreedyOptimizer(objective_function)

    selected_subset = optimizer.select(total_size=len(data_pool), budget=5)

    print(f"Selected Indices: {selected_subset}")
    print(f"Selected Data Points:\n{data_pool[selected_subset]}")

