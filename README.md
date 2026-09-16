# Sistema de recomendación de películas

Proyecto de aprendizaje que combina un ranking por votos, recomendaciones por contenido (TF-IDF) y un modelo colaborativo SVD. El filtro híbrido selecciona películas similares y las ordena según la nota estimada para un usuario conocido.

## Instalación

Se recomienda Python 3.11. Desde la carpeta del repositorio:

```bash
python -m venv .venv
```

Activa el entorno en Windows PowerShell con `.venv\Scripts\Activate.ps1`; en macOS/Linux con `source .venv/bin/activate`. Después:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`scikit-surprise` puede requerir herramientas de compilación de C/C++ según la plataforma. Se limita NumPy a 1.x para esta versión de Surprise. Los rangos de dependencias no son un lockfile; el entrenamiento completo debe verificarse en el entorno de destino.

## Datos necesarios

Descarga y descomprime [The Movies Dataset de Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Puede requerir iniciar sesión en Kaggle. Crea una carpeta `data` junto al cuaderno y coloca estos archivos con sus nombres originales:

- `data/movies_metadata.csv`
- `data/credits.csv`
- `data/ratings_small.csv`
- `data/links.csv`

No sustituyas `ratings_small.csv` por `ratings.csv`: el conjunto grande incrementaría considerablemente el coste de entrenamiento. Los datos no se distribuyen en este repositorio; consulta sus condiciones de uso en la fuente.

## Ejecución

Desde la raíz del proyecto:

```bash
python -m jupyterlab
```

Abre `Sistema de recomendación de películas.ipynb` y ejecuta **Restart Kernel and Run All Cells**. La búsqueda de parámetros realiza 324 combinaciones con tres particiones cada una, por lo que puede tardar. Las salidas antiguas se han eliminado para no presentarlas como resultados de esta versión.

El cuaderno permite:

- `get_genre_recommendations('Horror')`: ranking por género.
- `get_recommendations('The Ring')`: similitud de contenido.
- `improved_recommendations('The Ring')`: similitud, género y puntuación ponderada.
- `hybrid(100, 'Gattaca')`: recomendaciones para un usuario del dataset, excluyendo sus películas valoradas.

Las tres últimas funciones también aceptan el ID numérico de TMDB en lugar del título. Si un título identifica varias películas, se muestran los ID para elegir. Un usuario desconocido recibe un mensaje explícito; utiliza la recomendación por contenido cuando no hay historial.

## Evaluación y límites

Se reserva un 20 % de valoraciones para evaluación. La selección de parámetros usa solo el 80 % restante y se compara SVD con BaselineOnly sobre el mismo test. Después se entrena un modelo final con todas las valoraciones para recomendar. La muestra, las particiones y SVD utilizan una semilla fija.

RMSE y MAE evalúan predicciones de notas, no la calidad del ranking híbrido. Queda pendiente evaluar Precision@10/NDCG@10 y cobertura con datos reservados. La división aleatoria tampoco simula necesariamente recomendaciones futuras; para ese objetivo habría que evaluar una división temporal.

Se mantiene el filtro original del percentil 95 de votos: el catálogo favorece películas populares y no incluye todas las películas del dataset. El híbrido reordena hasta 25 candidatas no vistas; las películas sin historial colaborativo pueden recibir estimaciones menos personalizadas. La similitud se almacena en una matriz densa: ampliar mucho el catálogo exigiría revisar su consumo de memoria.

## Comprobaciones sin descargar los datos

```bash
python -m unittest discover -s tests -v
```

Las pruebas ejecutan las funciones del propio cuaderno con datos sintéticos y cubren sinopsis, deduplicación, índices, títulos ambiguos, precisión de notas y exclusión de películas vistas. El predictor colaborativo se simula: estas pruebas no sustituyen la ejecución completa con Kaggle y Surprise.

El PDF incluido corresponde a la versión original y no se ha regenerado.

## Autor y licencia

Albert Vila. Licencia MIT: [LICENSE.txt](LICENSE.txt).
