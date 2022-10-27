

# Скачивание файлов

Файлы поместить в папку content

Либо через [гугл диск](https://drive.google.com/drive/folders/1w-7hjwjGFDUxjbOibPui1lJ8TpM92I4G?usp=sharing)

Либо через wget

Датасет по профессионалам
```
wget --quiet https://daniilak.ru/cheb/train_proff.csv
wget --quiet https://daniilak.ru/cheb/validate_proff.csv
wget --quiet https://daniilak.ru/cheb/test_proff.csv
```
Датасет по студентам
```
wget --quiet https://daniilak.ru/cheb/train_student.csv
wget --quiet https://daniilak.ru/cheb/validate_student.csv
wget --quiet https://daniilak.ru/cheb/test_student.csv
```

# Пакеты pip

```
python3 -m pip install pandas numpy scikit-learn tqdm transformers torch torchmetrics ipdb
``` 

# Запуск обучения по моделям

```
python3 main.py
``` 

# Запуск обучения по CNN

```
python3 cnn.py
``` 

# Colab'ы


* Bert (cointegrated/rubert-tiny) [Студенты](https://colab.research.google.com/drive/1MQEruXVEPa174t_jFc-jlEwUdCuQ_XVd?usp=sharing), [Профессионалы](https://colab.research.google.com/drive/1itEu_bZ1z4N43VlU1jLAFmm3A1vQVRKa?usp=sharing)

* CNN [Студенты](https://colab.research.google.com/drive/1Gjb3GrjoDjgYbJz0ul3tN38noEzdVQq6?usp=sharing) [Профессионалы](https://colab.research.google.com/drive/1AUq3TV4-Ke3kijIIrVf7dwYA-EDVZrEh?usp=sharing)

* LaBSE-en-ru [Студенты](https://colab.research.google.com/drive/125KrQbz8twsvzz9iZLZGUon7IEfmPDnZ?usp=sharing)

