import numpy as np
from scipy.stats import entropy

from .metrics import Agreement


def semantic_clustering(
    inp,
    outs,
    agreement_fn,
    threshold=0.5,
):
    """
    Organizes similar items from a list into clusters based on a similarity threshold.

    This function takes a list of items (`outs`) and groups them into clusters.
    Each cluster contains items that are similar to each other based on a given similarity
    measure (`agreement_fn`). The similarity is compared to a specified threshold to
    determine if items belong in the same cluster.

    Parameters:
        inp (str): A reference input used by `agreement_fn` to compare against items in `outs`.
        outs (List(str)): A list of items to be clustered.
        agreement_fn: A function that measures similarity between `inp` and each item in `outs`, and between items within `outs`. Returns a similarity score.
        threshold (float, optional): A float value representing the minimum similarity score requiredfor items to be considered as part of the same cluster. Default value is 0.5.

    Returns:
        (List): A list of clusters, where each cluster is a list of items from 'outs' that are similar to each other as per the 'agreement_fn' and above the 'threshold' value.

    Example:
        ```
        semantic_clustering(input_item, list_of_items, similarity_function)
        > [[item1, item2], [item3], [item4, item5, item6]]
        ```
    """
    C = [[outs[0]]]
    outs = outs[1:]
    for i in range(len(outs)):
        STORED = False
        for j in range(len(C)):
            s_c = C[j][0]
            left_score = agreement_fn(inp, s_c, outs[i])
            right_score = agreement_fn(inp, outs[i], s_c)

            if left_score > threshold and right_score > threshold:
                STORED = True
                C[j].append(outs[i])
        if not STORED:
            C.append([outs[i]])
    return C


class ConsistencyScorer:
    """
    A class to calculate consistency scores for a set of outputs based on a given input.

    Attributes:
        - agreements_list (list): A list of tuples where each tuple contains an agreement function name and a threshold.
        - scoring_type (str): Type of scoring method to use. Options are 'entropy' or 'pairwise'.
        - aux_model: An auxiliary model used for calculating agreement scores.

    Methods:
        - entropy_score: Calculates the entropy score for a given set of outputs.
        - pairwise_score: Calculates the pairwise score for a given set of outputs.
        - score: Calculates the consistency score based on the specified scoring type.
    """

    def __init__(self, agreements_list, scoring_type, aux_model):
        """
        Initializes the ConsistencyScorer with necessary parameters.

        Parameters:
            - agreements_list (list): A list of agreement functions and thresholds.
            - scoring_type (str): The type of scoring to be used.
            - aux_model: An auxiliary model for agreement calculations.
        """
        super(ConsistencyScorer, self).__init__()
        self.agreements_list = agreements_list
        self.scoring_type = scoring_type
        self.aux_model = aux_model

    def entropy_score(self, input, outputs, agreement_fn, threshold, _):
        """
        Calculates the entropy score for given outputs based on semantic clustering.

        Parameters:
            - input: The input based on which outputs were generated.
            - outputs (list): A list of generated outputs.
            - agreement_fn: A function to calculate agreement between outputs.
            - threshold: A threshold value for agreement.

        Returns:
            - float: The entropy score.
        """
        # TODO
        # Add exact score via entropy estimate through Monte Carlo
        clusters = semantic_clustering(input, outputs, agreement_fn, threshold)

        pk = np.array([len(c) for c in clusters]) / sum([len(c) for c in clusters])
        H = entropy(pk, base=2)
        return H

    def pairwise_score(self, input, outputs, agreement_fn, threshold, binary=True):
        """
        Calculates pairwise agreement score for given outputs.

        Parameters:
            - input: The input based on which outputs were generated.
            - outputs (list): A list of generated outputs.
            - agreement_fn: Function to calculate agreement between two outputs.
            - threshold: A threshold value for considering an agreement.
            - binary (bool): If True, counts binary agreements; else adds up agreement scores.

        Returns:
            - float: The pairwise agreement score.
        """
        agreements = 0
        for i, output_i in enumerate(outputs):
            for j, output_j in enumerate(outputs):
                if i == j:
                    continue
                agreement_score = agreement_fn(input, output_i, output_j)
                if binary and agreement_score >= threshold:
                    agreements += 1
                elif binary == False:
                    agreements += agreement_score
        if (len(outputs) * (len(outputs) - 1)) == 0:
            return 0
        return (1 / (len(outputs) * (len(outputs) - 1))) * agreements

    def score(self, input, outputs):
        """
        Calculates the consistency scores for the given outputs based on the scoring type.

        Parameters:
            - input: The input based on which outputs were generated.
            - outputs (list): A list of generated outputs.

        Returns:
            - dict: A dictionary of scores for each agreement function in agreements_list.
        """
        if self.scoring_type == "entropy":
            scorer = self.entropy_score
        elif self.scoring_type == "pairwise":
            scorer = self.pairwise_score
        else:
            raise Exception(f"scoring type '{self.scoring_type}' not available")

        con_scores = {}
        for name, threshold in self.agreements_list:
            fn = Agreement(name, self.aux_model).agreement_fn
            print("Getting score for ", name)
            con_scores[name] = scorer(input, outputs, fn, threshold, False)
        return con_scores
