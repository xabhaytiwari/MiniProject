from abc import ABC, abstractclassmethod
from typing import List, Any

class SubModularFunction(ABC):
    """
    Abstract Base Class defining the structure of Submodular Functions.

    This class servers as a strict contract for all submodular functions implementations.
    """

    @ABC
    def evaluate(self, selected_set_indices: List[int]) -> float:
        """
        Calculates the total score of a set.

        :param selectedSet: The Selected Set, for which we're calculating the score.
        :type selectedSet: List
        :return: Return the Score for the set.
        :rtype: float
        """
        pass

    @ABC
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
    