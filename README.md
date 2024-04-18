# Sistema de Recomendación de Películas

## Descripción
Este proyecto desarrolla un sistema de recomendación de películas que utiliza técnicas de aprendizaje automático y filtrado colaborativo para proporcionar recomendaciones personalizadas a los usuarios. Basado en los datos de películas y calificaciones de usuarios de Kaggle, este sistema intenta mejorar la experiencia del usuario en plataformas de streaming, aumentando la retención y satisfacción del usuario.

## Características
- **Preprocesamiento de Datos**: Limpieza y transformación de conjuntos de datos para eliminar duplicados y manejar valores nulos.
- **Exploración de Datos**: Análisis de las características de las películas y calificaciones para entender mejor la distribución y las tendencias.
- **Filtrado Colaborativo**: Implementación del algoritmo SVD para predecir cómo los usuarios podrían calificar películas que no han visto aún.
- **Modelo de Recomendación basado en Contenido**: Uso de TfidfVectorizer para convertir texto a vectores y calcular similitudes usando linear_kernel.

## Tecnologías Utilizadas
- Python
- Pandas, Numpy
- Scikit-learn
- Matplotlib, Seaborn
- SVD

## Instalación
Para ejecutar este proyecto, clona el repositorio y asegúrate de tener instaladas las siguientes dependencias:
```bash
git clone https://github.com/albertvila78/Proyecto-final
cd Proyecto-final
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Contribuir
Las contribuciones son bienvenidas. Si deseas contribuir, por favor haz un fork del repositorio y crea un Pull Request con tus mejoras.

## Licencia
Este proyecto está bajo la Licencia MIT - vea el archivo LICENSE para más detalles.

## Autores
- **Albert Vila** - *Trabajo Inicial* - [AlbertVila](https://github.com/albertvila78)

## Referencias
- [The Movies Dataset en Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- [Aprendiendo Machine Learning](https://www.aprendemachinelearning.com/sistemas-de-recomendacion/)

