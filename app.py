from fastapi import FastAPI
from typing import List

from schema import PostGet
from datetime import datetime
import os
import pandas as pd
from sqlalchemy import create_engine, URL
from catboost import CatBoostClassifier
from sqlalchemy.orm import sessionmaker
from loguru import logger
from dotenv import load_dotenv

# Создадим подключение к базе данных через SQLAlchemy

SQLALCHEMY_DATABASE_URL = URL.create("postgresql",
				     username=os.environ["POSTGRES_USER"],
				     password=os.environ["POSTGRES_PASSWORD"],
    				     host=os.environ["POSTGRES_HOST"],
				     port=os.environ["POSTGRES_PORT"],
    				     database=os.environ["POSTGRES_DATABASE"])

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()


# Функция для получения пути до модели в зависимости от того локально или удаленно запускается сервис

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model_{model_version}'
    else:
        MODEL_PATH = path
    return MODEL_PATH


# Функция загрузки модели

def load_models():
    model_path = get_model_path("/Users/slava/ML/Project/catboost_model_ML")
    from_file = CatBoostClassifier()
    return from_file.load_model(model_path)


# Функция для выгрузки таблиц из БД по частям для экономии памяти

def batch_load_sql(query: str, engine) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# Функция выгрузки таблиц

def load_features(database_url):
    # Нам понадобится выгрузить все пары post_id, user_id где было действие like, чтобы проверять
    # и не рекомендовать пользователю уже лайкнутый им пост
    logger.info("Loading liked posts")
    liked_posts_query = """
    SELECT DISTINCT post_id, user_id
    FROM public.feed_data
    WHERE action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    logger.info("Loading posts features")
    post_features = pd.read_sql('SELECT * FROM syvl1526_post_features_lesson_22',
                                con=database_url)

    logger.info("Loading user features")
    user_features = pd.read_sql('SELECT * FROM public.user_data',
                                con= database_url)
    return [liked_posts, post_features, user_features]


# Загружаем модель, логируем

logger.info("Loading model")
model = load_models()


# Загружаем таблицы, логируем

logger.info("Loading features")
features = load_features(SQLALCHEMY_DATABASE_URL)

logger.info("Service is up and running")


# Напишем get запрос ответом которого будет список из id постов наиболее вероятных для лайка пользователем
# id которого подается в качестве params в том числе

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 10) -> List[PostGet]:
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    # Отберем фичи соответсвующего запросу юзера из стянутых из БД таблиц
    user_features = features[2].loc[features[2].user_id == id]
    # Дропаем id так как он не участвовал в построении модели
    user_features = user_features.drop('user_id', axis=1)

    logger.info("dropping columns")
    # Положим все посты со сгенерированными фичами в post_features
    post_features = features[1].drop('text', axis=1)
    # Выделим отдельно для удобства формирования ответа по требуемому типу датафрейм content
    content = features[1][['post_id', 'text', 'topic']]

    # Объединим датафрейм с постами с фичами юзера
    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = post_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Добавим информацию о времени
    logger.info("add time info")
    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    # Дропнем отсутсвующие в модели колонки
    user_posts_features = user_posts_features.drop(['os', 'source'], axis=1)

    # Предскажем вероятности лайка постов
    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # Удалим уже ранее лайкнутые юзером посты из датафрейма с предсказаниями
    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].index
    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ]


if __name__ == '__main__':
    load_dotenv()
    uvicorn.run(app)
