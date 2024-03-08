from bayes_utils import get_class_stats, get_gauss_pdf, calculate_accuracy
from numpy import ndarray

def predict(instance: ndarray, class_stats: dict):
    """
    Naive bayesin ana implementasyonudur.
    Veri setindeki her bir instance için bulunduğu classı tahmin eder.

    Algoritmanın açıklaması : 
    Her bir sınıf için verilen örneğin o sınıfa ait olup olmadığı,
    o sınıfın hesaplanan istatistikleri baz alınarak bulunur ve maksimum olasılığa sahip olan sınıf döndürülür
    """
    probs = {}
    for label, stats in class_stats.items():
        probs[label] = 1 # initilization
        for i, feature_value in enumerate(instance[:-1]):
            mean = stats['mean'][i]
            std = stats['std'][i]
            prob = get_gauss_pdf(feature_value, mean, std)
            probs[label] *= prob
    return max(probs, key=probs.get)


def train(train_data): 
    """
    Naive bayes için labelların istatistikleri toplar yani modeli birnevi eğitir.
    """
    return get_class_stats(train_data)

def eval(class_stats, val_data):
    """
    Verilen değerlendirme verisi üzerinde Naive Bayesin doğruluk oranını değerlendirir.
    """
    preds = [predict(instance, class_stats) for instance in val_data.to_numpy()]
    ground_truths = val_data.iloc[:, -1].tolist()

    return calculate_accuracy(ground_truths, preds)