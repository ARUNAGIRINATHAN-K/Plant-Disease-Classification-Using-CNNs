# Plant Disease Classification Using CNNs

*Trained a CNN model to automatically identify diseases in plant leaves from images, which can help farmers detect crop issues early and reduce losses.*

---

## Dataset([view](kaggle.com/datasets/abdallahalidev/plantvillage-dataset))
```
import kagglehub
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
```

- ~54,000 images across 38 classes (healthy + diseased)
- Covers 14 crop species (tomato, potato, corn, apple, etc.)
- Available on Kaggle and TensorFlow Datasets