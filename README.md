<h1 align='center'> Proyecto Individual N°1</h1>

<h2 align='center'> Machine Learning Operations (MLOps)</h2>

<h2 align='center'>Jessica Leandra Velasco P, DATAPT05</h2>

---

## **`Tabla de Contenidos`**

- [Introducción](#introducción)
- [Desarrollo](#desarrollo)
    - [ETL](#exploración-transformación-y-carga-etl)
    - [EDA](#análisis-exploratorio-eda)
    - [Sistema de recomendación](#modelo-de-recomendación)
    - [Despliegue de la API](#despliegue-para-la-api)
- [Contacto](#contacto)


---

# Introducción

En este proyecto, llevaremos a cabo un estudio basado en Machine Learning Operations (MLOps). Este estudio se divide en tres etapas principales:

1. **Exploración y Transformación:** Se realizará un análisis exploratorio de los datos, incluyendo la exploración de distribuciones y detección de correlaciones y valores atípicos.

2. **Preparación de Datos:** Se prepararán los datos para comprender las relaciones entre las variables y construir modelos sobre ellos. También se crearán funciones para consultas a los datos, consumibles a través de una API.

3. **Modelado:** Se desarrollarán modelos de Machine Learning para entender relaciones y predecir correlaciones entre variables.

Los datos utilizados incluyen información sobre juegos en la plataforma Steam y la interacción de los usuarios con estos juegos.

## Diccionario de los Datos

<p align="center"><img src="./images/Diccionario.jpg"></p>

---

# Desarrollo

### Exploración, Transformación y Carga (ETL)

A partir de los 3 dataset proporcionados (steam_games, user_reviews y user_items) referentes a la plataforma de Steam, en primera instancia se realizó el proceso de limpieza de los datos.

#### `steam_games`

- Se eliminaron filas completamente nulas y se corrigieron duplicados en el ID.
- Se completaron nulos en los géneros a partir de los datos de tags.
- Se completaron los valores de precio cuando este tenía un formato erróneo y era Free To Play; además, se normalizó la columna a valores reales.
- Las variables nulas en precio (menos del 4%) se eliminaron; otras filas con valores nulos también se eliminaron al no poder hacer un tratamiento más profundo y ser una pequeña parte del dataset.
- Se crearon variables ficticias (dummies) en la columna género para el análisis.
- Se extrajeron años de la columna release_date, teniendo en cuenta los distintos formatos, y las filas donde no podía extraerse el año se eliminaron.
- Se eliminaron columnas no utilizadas.
- Se exportó para tener el dataset limpio.

#### `user_reviews`

- Se realizó un explode ya que la columna de review era una lista de diccionarios.
- Se eliminaron filas con valores nulos en la columna de "reviews".
- Se creó una nueva columna llamada 'sentiment_analysis' usando análisis de sentimiento y se eliminó la columna de review.
- Se exportó para tener el dataset limpio.

#### `user_items`

- Se realizó un explode ya que la columna de items era una lista de diccionarios.
- Se eliminaron filas con valores nulos en la columna de "items".
- Se exportó para tener el dataset limpio.

### Análisis Exploratorio (EDA)

Teniendo los 3 dataset limpios, se realizó un proceso de EDA para realizar gráficos y así entender las estadísticas, encontrar valores atípicos y orientar un futuro análisis.

#### `steam_games`

- Primero se encontró la distribución de los precios a partir de un gráfico de cajas y bigotes, encontrando muchos valores atípicos. Sin embargo, considerando el contexto, no son valores necesariamente erróneos, ya que se pueden encontrar juegos de centavos de dólar y juegos de miles de dólares. Los últimos son los menos usuales.
- Se hizo un gráfico de barras con la distribución de juegos por año, incluyendo los contenidos Free. Se encontró que el año 2015 tuvo la mayor cantidad de juegos y la mayor cantidad de Free.

#### `user_reviews`

- Se realizó un gráfico de barras con la cantidad de sentimientos positivos y de estos, los que recomiendan. El resultado mostró que hubo muchos sentimientos positivos seguidos de los neutrales. Además, se observó que un porcentaje de los sentimientos positivos no recomiendan y en los sentimientos negativos, un porcentaje sí recomienda. Esto podría deberse a alguna falla en el análisis de sentimiento, sin embargo, es un porcentaje bajo.

#### `user_items`

- Para la columna playtime_forever, con un diagrama de cajas y bigotes, se analizó la distribución y se encontraron muchos valores atípicos. No obstante, a falta de un mejor análisis, no se realizará un tratamiento, ya que no necesariamente son errores; sin embargo, se debe verificar si hay algún valor de playtime_forever que para el ítem dado tenga más horas que el año de lanzamiento del ítem.
- En la verificación, se encontró que no hay ningún valor que cumpla estas condiciones, por lo que no se modificarán estos valores.
- También se calculó la dispersión entre playtime_forever y la cantidad de ítems.
- A partir de varias tablas, se graficaron los 15 juegos con más horas y los 15 desarrolladores con más horas en sus juegos.
- También se graficaron los desarrolladores con más recomendaciones positivas.

### Modelo de Recomendación

#### `Filtro Colaborativo`

- A partir de la tabla de user_reviews, se utilizaron las columnas user_id, item_id, recommend y sentiment_analysis. A partir de las dos últimas, se generó una nueva columna llamada 'rating', la cual tiene una escala entre 0 y 5. Se utilizó la técnica de Descomposición de Valor Singular (SVD) para realizar un filtro colaborativo en función de estas 3 columnas. Se utilizó GridSearch para elegir hiperparámetros óptimos; el modelo final obtuvo un RMSE de 0.85. En una escala de 0 a 5, una desviación de 0.85 es un resultado aceptable, teniendo en cuenta que el ranking podría haberse elegido de manera más óptima. EL modelo se exportó como pkl para posteriormete ser consumido por la API a través de la función<br>

Puntos a mejorar: Se podría haber elegido otro modelo como KNNBasic. Además, el proceso de generación de ratings podría haberse optimizado. También es importante considerar que un filtro basado únicamente en estas características podría estar sesgado, ya que los usuarios tienden a opinar más sobre productos que no les gustaron de forma negativa. Para abordar este problema, se podría complementar con un perfil de usuario y explorar similitudes entre ellos.

#### `Filtro basado en Contenido`

- Usando la tabla de steam_games y tomando las columnas de géneros como dummies, NearestNeighbors, el cual se encarga de buscar los k vecinos mas cercanos. Puntos a mejorar para este filtro: usar otras columnas como desarrollador o especificaciones del juego. Además, no se pudo verificar adecuadamente la performance del modelo.

Como extra en cuanto al filtro colaborativo, al usar la tabla de revisiones de usuarios, si este no se encontraba, no podía hacer recomendaciones. Por esto, para aquellos casos, se buscaba al usuario en la tabla de ítems y se realizaba una recomendación con el filtro basado en contenido en función del ítem en el que tuviera más horas y que no estuviera en las revisiones.

En general, se obtuvieron modelos aceptables; sin embargo, con un análisis más profundo, podrían haberse obtenido mejores resultados. Algunas mejoras podrían incluir la alteración ponderada, la fusión de resultados o incluso un **modelo híbrido complejo**.

### Despliegue para la API

Se desarrollaron las siguientes funciones, a las cuales se podrá acceder desde la API en la página Render:

- **`developer(desarrollador: str)`**: Retorna la cantidad de ítems y el porcentaje de contenido gratis por año para un desarrollador dado.
- **`userdata(User_id: str)`**: Retorna el dinero gastado, cantidad de ítems y el porcentaje de comentarios positivos en la revisión para un usuario dado.
- **`UserForGenre(género: str)`**: Retorna al usuario que acumula más horas para un género dado y la cantidad de horas por año.
- **`best_developer_year(año: int)`**: Retorna los tres desarrolladores con más juegos recomendados por usuarios para un año dado.
- **`developer_rec(desarrolladora: str)`**: Retorna una lista con la cantidad de usuarios con análisis de sentimiento positivo y negativo para un desarrollador dado.
- **`ser_recommend(user:str)`**: Esta función recomienda 5 juegos para un usuario especificado usando un filtro colaborativo.
- **`item_recommend(item:int)`**: Esta función recomienda 5 ítems dado un ítem específico usando un filtro basado en contenido.




 


