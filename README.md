# Разработка сервиса удаления дубликатов и классификации новостей
Сервис будет удалять дубликаты новостей из фиксированного набора и классифицировать новости по заранее известным категориям. В случае идентичной тематики в дублях будет выбираться та, которая более насыщена целевой информацией

Сервис был разработан в рамках хакатона [AI New](https://codenrock.com/contests/ai-news#/)


## Запуск контейнера

`docker build --tag 'news_classification' . `<br>
`docker run --rm -p 8000:8000 news_classification`

После запуска контейнера документация по API доступна по [http://localhost:8000/docs]()

Системные требования:

- используется многопоточная модель, количество ядер CPU имеет значение
- проверки производились с RAM 8Gb (оценка потребности модели до 2Gb, поэтому должно быть достаточно 4Gb, но в таком режиме не тестировали)
- GPU не используется

Просьба связаться, в случае вопросов или возникнования проблем. 


## Методы API
- /health - всегда возвращает OK
- /process-csv/ - принимает файл csv и возвращает обработанный файл

Файл csv должен иметь колонку `text`. При наличии колонки `channel_id` она будет возвращена в результирующем файле. 

Результирующий файл имеет колоки `text`, `category` и `channel_id` при её наличии во входном файле. Выходной файл очищен от дубликатов и может содержать меньшее количество строк.

Примеры входных файлов находятся в папке `datasets`. Примеры выходных файлов в папке `datasets/results`

## Запуск вне контейнера

Для многопоточной обработки используется библиотека joblib с параметром `backend="multiprocessing"` в файле `duplicates_processor.py`. Однако в контейнере этот тип работал не стабильно (после обработки одного файла останавливался контейнер). По этой причине изменено на `threading`, что отразилось на скорости.

Так как вне контейнера многозадачность работает корректно и время выполнения значительно меньше, то приводим код запуска вне контейнера. Предварительно нужно изменить `backend="multiprocessing"`

`cd .`<br>
`python -m venv venv`<br>
`source venv/bin/activate`<br>
`pip install -r requirements.txt`<br>
`export PYTHONPATH=$PYTHONPATH:$PWD`<br>
`python app/app.py`<br>

В таком режиме полная обработка 50k строк занимала менее 4-ех минут.

## Презентация
Описание работы алгоритмов и многое другое вы можете найти в нашей [презентации](https://github.com/dmitrii-naumenko/news-classification/blob/main/presentation/news-classification.pdf) 



## Возможности для оптимизации
- как так по факту удаление дублей более длительная операция в этой реализации и имеет квадратичную сложность, то можно удалять дубликаты после классификации отдельно в каждом классе. Это уменьшит общее время выполнения.
- подбор гиперпараметров и описаний категорий при наличии качественно размеченного датасета
- использование дополнительной информации, например номера канала и коррекция весов классов в соотвествии с распределением классов в каждом канале
- создание нескольких простых эвристических правил с проверкой на размеченном датасете

## Команда

- Науменко Дмитрий https://t.me/naumenko_ds
- Кутькина Татьяна https://t.me/Tatyanna_Kutkina
- Дамдинов Зорикто https://t.me/suzuyajxiii

