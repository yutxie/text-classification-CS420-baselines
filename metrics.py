import torch


class CategoricalAccuracy():
    """
    https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/categorical_accuracy.py#L11-L103
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predsicted scores.
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ValueError("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise ValueError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predsictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                #  mask: Optional[torch.Tensor] = None):
                 mask=None):
        """
        Parameters
        ----------
        predsictions : ``torch.Tensor``, required.
            A tensor of predsictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predsictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # predsictions, gold_labels, mask = self.unwrap_to_tensors(predsictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predsictions.size(-1)
        if gold_labels.dim() != predsictions.dim() - 1:
            raise ValueError("gold_labels must have dimension == predsictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predsictions.size()))
        if (gold_labels >= num_classes).any():
            raise ValueError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        predsictions = predsictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predsictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predsictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predsictions.topk(min(self._top_k, predsictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # predsiction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predsictions = predsictions.max(-1)[0]
            max_predsictions_mask = predsictions.eq(max_predsictions.unsqueeze(-1))
            # max_predsictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predsictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predsictions_mask[torch.arange(gold_labels.numel()).long(), gold_labels].float()
            tie_counts = max_predsictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


class F1Measure():
    """
    https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/f1_measure.py#L10-L94
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int = 1) -> None:
        self._positive_label = positive_label
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predsictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                #  mask: Optional[torch.Tensor] = None):
                 mask=None):
        """
        Parameters
        ----------
        predsictions : ``torch.Tensor``, required.
            A tensor of predsictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predsictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # predsictions, gold_labels, mask = self.unwrap_to_tensors(predsictions, gold_labels, mask)

        num_classes = predsictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ValueError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask

        argmax_predsictions = predsictions.max(-1)[1].float().squeeze(-1)

        # True Negatives: correct non-positive predsictions.
        correct_null_predsictions = (argmax_predsictions !=
                                    self._positive_label).float() * negative_label_mask
        self._true_negatives += (correct_null_predsictions.float() * mask).sum()

        # True Positives: correct positively labeled predsictions.
        correct_non_null_predsictions = (argmax_predsictions ==
                                        self._positive_label).float() * positive_label_mask
        self._true_positives += (correct_non_null_predsictions * mask).sum()

        # False Negatives: incorrect negatively labeled predsictions.
        incorrect_null_predsictions = (argmax_predsictions !=
                                      self._positive_label).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predsictions * mask).sum()

        # False Positives: incorrect positively labeled predsictions
        incorrect_non_null_predsictions = (argmax_predsictions ==
                                          self._positive_label).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predsictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0


class Metrics():

    def __init__(self):
        self.metrics = [CategoricalAccuracy(), F1Measure()]

    def count(self, preds, targs):
        for metric in self.metrics:
            metric(preds, targs)

    def report(self, reset=False):
        report = []
        for metric in self.metrics:
            _ = metric.get_metric(reset=False)
            if isinstance(metric, CategoricalAccuracy): report.append(('acc', _))
            elif isinstance(metric, F1Measure): report += [('rec', _[0]), ('pre', _[1]), ('f1', _[2])]
            else: raise NotImplementedError
        return report

    def reset(self):
        for metric in self.metrics:
            metric.reset()