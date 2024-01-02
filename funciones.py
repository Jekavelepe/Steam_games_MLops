import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from surprise import SVD
import numpy as np

'''FUNCION_1'''

def PlayTimeGenre(genero: str):
    '''Esta función debe devolver año con mas horas jugadas para dicho género.
    Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}'''
    
    # Leer el DataFrame desde el archivo CSV
    funtion_one_merge_df = pd.read_csv('./datasets/PlayTimeGenre.csv.gz')

    # Capitalizar la primera letra del género
    genero = genero.capitalize()

    # Filtrar el DataFrame para obtener solo las filas relacionadas con el género específico
    filtro_df = funtion_one_merge_df[funtion_one_merge_df['genres'].apply(lambda x: genero in x)]

    if not filtro_df.empty:
        # Agrupar por año sumando las horas jugadas para cada año
        grouped_df = filtro_df.groupby('year')['playtime_forever'].sum().reset_index()

        # Encontrar el año con la mayor cantidad de horas jugadas
        max_year = grouped_df.loc[grouped_df['playtime_forever'].idxmax()]['year']

        return {"Año de lanzamiento con más horas jugadas para Género {}:".format(genero): int(max_year)}
    else:
        return {"Género {} no encontrado en el DataFrame.".format(genero): None}
    
'''FUNCION_2'''

def UserForGenre(genero: str):

    '''Esta función devolver el usuario que acumula más horas jugadas para el género dado y una lista
    de la acumulación de horas jugadas por año.Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" :
      us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}'''

    # Leer el DataFrame desde el archivo CSV
    funtion_two_merge_df = pd.read_csv('./datasets/UserForGenre.csv.gz')

    genero = genero.capitalize()
    # Paso 1: Filtrar el DataFrame para el género dado
    filtered_df = funtion_two_merge_df[funtion_two_merge_df['genres'].apply(lambda x: genero in x)]

    if not filtered_df.empty:
        # Paso 2: Encontrar el usuario con más horas jugadas
        user_with_max_hours = filtered_df.groupby('user_id')['playtime_forever'].sum().idxmax()

        # Paso 3: Crear una lista de acumulación de horas jugadas por año
        grouped_df = filtered_df.groupby(['year', 'user_id'])['playtime_forever'].sum().reset_index()
        hours_accumulation = [{'Año': int(year), 'Horas': int(hours)} for year, hours in grouped_df.groupby('year')['playtime_forever'].max().items()]

        # Paso 4: Retornar el resultado en el formato deseado
        result = {"Usuario con más horas jugadas para Género {}".format(genero): user_with_max_hours,
                  "Horas jugadas": hours_accumulation}
        return result
    else:
        return {"Género {} no encontrado en el DataFrame.".format(genero): None}  
    
'''Funcion 3'''

def UsersRecommend(anio: int):

    '''Esta función el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = 
    True y comentarios positivos/neutrales) Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]'''

    user_reviews = pd.read_csv('./datasets/UsersRecommend_UsersNotRecommend.csv.gz')
   
    # Paso 1: Filtrar el DataFrame para el año dado
    reviews_anio = user_reviews[user_reviews['year'] == anio]
    
    if reviews_anio.empty:
        return {"mensaje": "No hay revisiones para el año proporcionado."}

    # Paso 2: Filtrar para incluir solo revisiones recomendadas y con sentimiento positivo o neutral
    relevant_reviews = reviews_anio[(reviews_anio['recommend'] == True) & (reviews_anio['sentiment_analysis'].isin([1, 2]))]

    if relevant_reviews.empty:
        return {"mensaje": "No hay revisiones recomendadas y con sentimiento positivo o neutral para el año proporcionado."}

    # Paso 3: Contar la cantidad de recomendaciones para cada juego
    game_recommendation_counts = relevant_reviews.groupby('title')['recommend'].count().reset_index()

    # Paso 4: Ordenar los juegos en función de la cantidad de recomendaciones
    top_games = game_recommendation_counts.sort_values(by='recommend', ascending=False)

    # Paso 5: Seleccionar los tres juegos principales
    top_3_games = top_games.head(3)

    # Paso 6: Crear el formato de retorno deseado
    result = [{"Puesto {}: {}".format(i + 1, title)} for i, title in enumerate(top_3_games['title'][:3])]

    return result

'''Funcion 4'''

def UsersNotRecommend(anio: int):

    '''Esta función el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)    
    Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]'''

    user_reviews = pd.read_csv('./datasets/UsersRecommend_UsersNotRecommend.csv.gz')
   
    # Paso 1: Filtrar el DataFrame para el año dado
    reviews_anio = user_reviews[user_reviews['year'] == anio]
    
    if reviews_anio.empty:
        return {"mensaje": "No hay revisiones para el año proporcionado."}

    # Paso 2: Filtrar para incluir solo revisiones no recomendadas y con sentimiento negativos
    relevant_reviews = reviews_anio[(reviews_anio['recommend'] == False) & (reviews_anio['sentiment_analysis'] == 0)]

    if relevant_reviews.empty:
        return {"mensaje": "No hay revisiones no recomendadas y con sentimiento negativos para el año proporcionado."}

    # Paso 3: Contar la cantidad de recomendaciones negativas para cada juego
    game_recommendation_counts = relevant_reviews.groupby('title')['recommend'].count().reset_index()

    # Paso 4: Ordenar los juegos en función de la cantidad de recomendaciones
    top_games = game_recommendation_counts.sort_values(by='recommend', ascending=False)

    # Paso 5: Seleccionar los tres juegos principales
    top_3_games = top_games.head(3)

    # Paso 6: Crear el formato de retorno deseado
    result = [{"Puesto {}: {}".format(i + 1, title)} for i, title in enumerate(top_3_games['title'][:3])]

    return result

'''Funcion 5'''

def sentiment_analysis(anio: int):
    '''Esta funcion según el año de lanzamiento, se devuelve una lista con la cantidad de 
    registros de reseñas de usuarios que se encuentren categorizados con un análisis de 
    sentimiento.Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}'''

    merge_df5 = pd.read_csv('./datasets/sentiment_analysis.csv.gz')

    # Filtrar el DataFrame por el año proporcionado
    relevant_reviews = merge_df5[merge_df5['year'] == anio]

    # Contar la cantidad de registros para cada categoría de análisis de sentimiento
    sentiment_counts = relevant_reviews.groupby('sentiment_analysis').size()

     # Convertir los índices a tipos de datos estándar (int)
    sentiment_counts.index = sentiment_counts.index.astype(int)

    # Crear el diccionario de resultados
    result = {
        'Negative': int(sentiment_counts.get(0, 2)),
        'Neutral': int(sentiment_counts.get(1, 2)),
        'Positive': int(sentiment_counts.get(2, 3))
    }
    

    return  result

'''Recomendaciones 1'''

def recomendacion_juego(id_producto):
    '''Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.'''

    if not isinstance(id_producto, int):
        try:
            id_producto = int(id_producto)
        except ValueError:
            return 'El Id debe ser un número entero'

    df_steam_games_with_dummies = pd.read_csv('./datasets/df_steam_games_dummies.csv.gz')
    steam_game_id_title = pd.read_csv('./datasets/steam_games_id_title.csv.gz')
    #Verificamos si el id ingresado esta en la base de datos
    if id_producto not in df_steam_games_with_dummies['id'].unique():
        return "ID no encontrado"
    #convertimos la columna id en index
    df_steam_games_with_dummies.set_index('id', inplace=True)

    # Obtener las características del juego dado su ID
    juego_caracteristicas = df_steam_games_with_dummies.loc[id_producto].values.reshape(1, -1)

    # Calcular la similitud del coseno entre el juego dado y todos los otros juegos
    similarities = cosine_similarity(df_steam_games_with_dummies.values, juego_caracteristicas)

    # Ordenar los juegos según su similitud y tomar los 6 juegos más similares (el primero es el mismo juego)
    similar_juegos_indices = similarities.flatten().argsort()[-6:-1][::-1]

    # Obtener los títulos de los juegos recomendados
    recommended_juegos = steam_game_id_title.loc[similar_juegos_indices, 'title'].tolist()

    return recommended_juegos


'''Recomendaciones 2'''

def recomendacion_usuario(id_usuario:str):
    ''' Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.'''
    #importamos los datasets
    df_ratings = pd.read_csv('./datasets/user_review_rating.csv.gz')
    steam_game_id_title = pd.read_csv('./datasets/steam_games_id_title.csv.gz')

    with open('./datasets/SVD_model.pkl', 'rb') as archivo:
        model = pickle.load(archivo)

    #Verificamos si el usuario esta en la base de datos
    if id_usuario not in df_ratings['user_id'].unique():
        return "ID no encontrado"
    
    # Obtener todos los juegos disponibles
    todos_los_juegos = df_ratings['item_id'].unique()

    # Obtener los juegos valorados por el usuario
    juegos_valorados_por_usuario = df_ratings[df_ratings['user_id'] == id_usuario]['item_id'].unique()

    # Obtener los juegos no valorados por el usuario
    juegos_no_valorados = list(set(todos_los_juegos) - set(juegos_valorados_por_usuario))

    # Crear un DataFrame con los juegos no valorados por el usuario
    df_juegos_no_valorados = pd.DataFrame(juegos_no_valorados, columns=['item_id'])

    # Hacer predicciones para los juegos no valorados por el usuario
    df_juegos_no_valorados['prediccion'] = df_juegos_no_valorados['item_id'].apply(lambda x: model.predict(id_usuario, x).est)

    # Ordenar los juegos por la calificación predicha y tomar los primeros n juegos como recomendación
    juegos_recomendados = df_juegos_no_valorados.sort_values(by='prediccion', ascending=False)['item_id'].tolist()

    # Obtener los títulos de los juegos recomendados que existen en steam_game_id_title
    recommended_juegos = []
    for juego_id in juegos_recomendados:
       juego_titulo = steam_game_id_title.loc[steam_game_id_title['id'] == juego_id, 'title'].tolist()
       if juego_titulo:
            recommended_juegos.append(juego_titulo[0])
            if len(recommended_juegos) == 5:
                break

    return recommended_juegos