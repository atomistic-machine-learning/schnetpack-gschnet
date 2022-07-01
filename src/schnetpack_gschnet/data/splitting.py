from typing import List
import operator
import torch
import logging
import numpy as np
from schnetpack.data import SplittingStrategy
from tqdm import tqdm

__all__ = ["FilteredRandomSplit"]


class FilteredRandomSplit(SplittingStrategy):
    """
    Splitting strategy that filters out some data points from the dataset and puts them
    into the test split, i.e. removes them from the training and validation data. The
    remaining structures are assigned uniformly randomly to the respective splits.
    """

    operators = {
        "!=": operator.ne,
        "==": operator.eq,
        "=": operator.eq,
        "<=": operator.le,
        "<": operator.lt,
        ">=": operator.ge,
        ">": operator.gt,
    }

    def __init__(self, filters, transforms=None):
        """
        Args:
            filters: List of filters used to exclude certain structures from the
                training and validation splits (e.g. certain compositions or molecules
                with specific property values). The excluded structures are included in
                the test split. Each molecule that matches at least one of the filters
                in the list is excluded (i.e. we take the union of the result of each
                individual filter to obtain the set of excluded structures).
                Each filter is a dictionary with entries `property`, `operator`, and
                `value` and we check if `datapoint[property] operator value`, e.g.
                datapoint[gap] < 4.0 to filter structures with gap smaller than four.
                The entries can also be lists of properties, operators, and values.
                Then, the filter only evaluates to True if all conditions specified
                in the lists are matched by the molecule (e.g. ["gap", "energy"],
                ["<", ">"], [6.0, -11000.0] to filter all molecules with gap < 6.0 and
                energy > -11000.0).
            transforms: Preprocessing transform applied to each system before the
                filter is applied (e.g. to extract the composition).
        """
        super().__init__()
        self.filters = filters
        self.transforms = transforms

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        logging.info(
            f"Filtering structures to excluded molecules from training and "
            f"validation splits according to the filters: "
            f"{self.filters}."
        )
        if len(split_sizes) != 3:
            raise ValueError(
                "Please specify all three split sizes (`num_train`, `num_val`, and "
                "`num_test` in that order)"
            )
        else:
            num_train, num_val, num_test = split_sizes
        # find molecules to exclude from train/val split using the filter
        excluded = np.zeros(len(dataset), dtype=bool)
        for i in tqdm(range(len(dataset))):
            # get data point
            inputs = dataset[i]
            # apply transforms
            if self.transforms is not None:
                for t in self.transforms:
                    inputs = t(inputs)
            for f in self.filters:
                prop = f["property"]
                op = f["operator"]
                value = f["value"]
                if not isinstance(op, str):
                    for j in range(len(op)):
                        exclude = True
                        if not self._apply_filter(op[j], inputs[prop[j]], value[j]):
                            exclude = False  # filter does not apply to data point
                            break
                else:
                    exclude = self._apply_filter(op, inputs[prop], value)
                if exclude:
                    excluded[i] = True
                    break

        # split data accordingly
        n_excluded = np.sum(excluded)
        n_included = len(dataset) - n_excluded
        logging.info(
            f"Excluded {n_excluded} structures that match the filters from the "
            f"training and validation splits."
        )
        if num_train + num_val > len(dataset) - n_excluded:
            raise ValueError(
                f"Your filters are too restrictive! {num_train} + {num_val}"
                f" data points are required for train + val splits, but only "
                f"{n_included} data points are available. {n_excluded} data "
                f"points were removed due to the filters: {self.filters}."
            )
        if num_test is None:
            num_test = n_included - num_train - num_val
        else:
            if num_test - n_excluded < 0:
                raise ValueError(
                    f"The selected test split size ({num_test}) is larger "
                    f"than the number of excluded molecules ({n_excluded}), which "
                    f"are supposed to be automatically assigned to the test split. "
                    f"Please increase the size of the test split."
                )
        lengths = [num_train, num_val, num_test]
        offsets = torch.cumsum(torch.tensor(lengths), dim=0)
        indices = np.random.permutation(np.nonzero(~excluded)[0]).tolist()
        train_idx, val_idx, test_idx = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, lengths)
        ]
        test_idx += np.nonzero(excluded)[0].tolist()  # add excluded to test split
        return [train_idx, val_idx, test_idx]

    def _apply_filter(self, op, val1, val2):
        op = self.operators[op]
        if isinstance(val1, torch.Tensor):
            val1 = val1.squeeze().tolist()
        if isinstance(val1, List):
            return all([op(v1, v2) for v1, v2 in zip(val1, val2)])
        else:
            return op(val1, val2)
