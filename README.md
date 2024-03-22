# naive_bayes_from_scratch

Naive bayes algoritması temel olarak, verilen bir örneğin belirli bir sınıfa ait olma olasılığını hesaplamak için bayes olasılığını kullanır. Modele herhangi bir "prior" bilgi verilmediğini kabul edersek **eğitim verisinden özniteliklerin ortalama ve varyans gibi değerlerini alarak bir normal dağılım olasılık fonksiyonu hazırlar ve diğer örnekleri bu oluşturulan olasılık dağılımından yararlanarak sınıflandırır.** "Naive" (saf) olarak adlandırılmasının nedeni, her özelliğin birbirinden bağımsız olduğu varsayımına dayanmasıdır, yani özellikler arasında bir ilişki olmadığı kabul edilir. Naive Bayes algoritması, genellikle metin sınıflandırması gibi doğal dil işleme problemlerinde ve özellikle spam filtreleme gibi uygulamalarda kullanılır. Ancak tabii ki herhangi sınıflandırma işini yapabilir. Bu kütüphane ise öğrenme amaçlı olup naive bayes algoritmasının sıfırdan minimal inşa edilimidir. Bu implementasyon genel olarak saf Python kodundan ibaret olsada NumPy, Pandas ve SciPy gibi standart kütüphanelerin bazı fonksiyonları veri yükleme, olasılık hesaplama vs. gibi temel işlemler için kullanılmıştır.

Kütüphanenin Temel Kullanımı :

```python 
from naive_bayes import train, eval

# train_data ve val_data bir pandas dataframe'i olmalıdır ve son sütunlarında label sütunu(nümerik veya string değerler olabilir) bulunmalıdır.
train_data = pd.read_csv("train.csv")
val_data = pd.read_csv("val.csv")

class_stats = train(train_data)
acc = eval(class_stats, val_data, var_smoothing=1e-10)
print(f"Doğruluk oranı: {acc}")
```

Modelin tek hiperparametresi temelde sıfıra bölüm hatasını engellemek için kullanılan var_smoothing değeridir. Default değer genel olarak iyi çalışmaktadır ve 1e-7, 1e-10 gibi diğer düşük değerler modelin performansına genelde bir etki etmemektedir ancak büyük değerler(1, 10 gibi) modelin performansını önemli ölçüde düşürebilir.

Kütüphanenin kullanımına dahil daha detaylı bir örnek diabetes.csv veriseti üzerinde [example_nb](https://github.com/baranbingol1/naive_bayes_from_scratch/blob/main/example/example_nb.ipynb) dosyasında verilmiştir.

Dosyaların açıklamaları :

bayes_utils.py : Algoritma için gerekli olan bazı gerekli fonksiyonlar bulunur.\
naive_bayes.py : Algoritmanın ana implemantasyonunu(predict fonksiyonu) ve veri seti üzerinde kullanmak için high-level iki fonksiyon içerir(train ve eval).
