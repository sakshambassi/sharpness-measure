import numpy as np
import torch

class Margin:
    def __init__(self):
        pass
    
    def calculate(self, 
                  probabilities: torch.Tensor, 
                  true_label_indexes: list, 
                  margin_type: str = 'mean'
                  ) -> float:
        """
        computes margin of a model based on the given probabilities

            mean_margin = mean_i^n(prob(y_i) - max_prob_i)
                where max_prob_i is max prob for any label != y_i
    
            min_margin = mean_i^n(prob(y_i) - max_prob_i)
                where max_prob_i is max prob for any label != y_i

        Args:
            probabilities (torch.Tensor): output of nn.functional.softmax(x.logits, dim=-1)
            true_label_indexes (like [0,1,2,...]): list of indexes of true labels in the input
            margin_type (str): either 'mean' to return mean margin else 'min' to return min_margin
        Returns:
            float: minimum margin or average margin
        """
        if margin_type == 'min': min_margin = float("inf")
        prob_count, total_margin = 0, 0
        for i, prob in enumerate(probabilities):
            # updating prob_count
            prob_count += 1
            # predicted prob of label
            true_label_prob = prob[true_label_indexes[i]].item()
            # setting predicted prob of label to -1 to calculate max
            prob[true_label_indexes[i]] = -1
            # calculating max_pred_difference for y != label
            max_pred_difference = (true_label_prob - torch.max(prob))
            # updating min_margin
            if margin_type == 'min': min_margin = min(min_margin, max_pred_difference)
            # updating total margin value for mean margin
            total_margin += max_pred_difference.item()
        # compute the average margin
        mean_margin = total_margin / prob_count

        if margin_type == 'min':
            return min_margin
        else:
            return mean_margin
