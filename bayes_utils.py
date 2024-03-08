import numpy as np
import pandas as pd
from scipy.stats import norm

def get_class_stats(dataset: pd.DataFrame) -> dict:
    """
    Veri setini Pandas DataFrame objesi olarak alır ve her bir ayrı classın istatistiklerini hesaplar.
    Class etiketinin dataframe içinde son column(sütun)da olması gerekir.
    Her bir label değerinin istatistiklerini içeren bir dictionary objesi döndürür.
    """
    class_stats = {}

    label_name = dataset.columns[-1]

    for label_value in dataset[label_name].unique():
        class_data = dataset[dataset[label_name] == label_value]
        # mean ve std değerleri her bir feature için bir değer döndürür dolayısıyla bir vektördür(np.ndarray objesi).
        stats = {
            "mean": np.mean(class_data, axis=0),
            "std": np.std(class_data, axis=0),
            "length": len(class_data)
        }
        class_stats[label_value] = stats

    return class_stats

def get_gauss_pdf(x, mean, std): return norm.pdf(x, loc=mean, scale=std)

def calculate_accuracy(ground_truths, preds):
    """
    Doğruluk oranını hesaplar ve geri döndürür.
    """
    correct_preds = 0
    for ground_truth, pred in zip(ground_truths, preds):
        if ground_truth == pred:
            correct_preds += 1
    return correct_preds / len(ground_truths)