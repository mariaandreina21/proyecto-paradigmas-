import streamlit as st
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import io
from fpdf import FPDF
import base64



def download_dataset(dataset_name):
    """
    Descarga un dataset de Kaggle
    dataset_name debe estar en formato 'usuario/nombre-dataset'
    *Funciona para descargar cualquier dataset desde kaggle siempre que pueda
    hacer la autenticacion del api de kaggle, este programa podra trabajar con cualquier
    dataset hasta el grafico de correlacion, posteriormente esta diseñado para el EDA y 
    Modelo de ML del dataset precargado en el ejemplo
    """
    try:
        # Configurar la API de Kaggle
        api = KaggleApi()
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

        api = KaggleApi()
        api.authenticate()

        
        # Crear directorio para los datos si no existe
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Descargar el dataset
        api.dataset_download_files(dataset_name, path='data', unzip=True)
        return True
    except Exception as e:
        st.error(f"Error al descargar el dataset: {str(e)}")
        return False
        """
        Definicion de funcion para exportar en pdf los resultados mas relevantes para
        el usuario del dataset
        """   
def export_report(df_clean, stats_by_crim, stats_by_ptratio, stats_by_dis, 
                 stats_by_rm, stats_by_lstat, stats_by_age,  # Nuevos parámetros
                 feature_importance_reg, feature_importance_class, mse, r2, accuracy):
    pdf = FPDF()
    pdf.add_page()
    
    # Configuración de fuente
    pdf.set_font('Arial', 'B', 16)
    
    # Título
    pdf.cell(0, 10, 'Informe de Análisis de Precios de Viviendas', 0, 1, 'C')
    pdf.ln(10)
    
    # Información General
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Información General del Dataset', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Número total de registros: {len(df_clean)}', 0, 1, 'L')
    pdf.cell(0, 10, f'Número de características: {len(df_clean.columns)}', 0, 1, 'L')
    pdf.ln(5)
    
    # Estadísticas por Criminalidad
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Análisis por Nivel de Criminalidad', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_crim.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)
    
    # Estadísticas por Ratio Alumno-Profesor
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Análisis por Ratio Alumno-Profesor', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_ptratio.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)
    
    # Estadísticas por Distancia
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '4. Análisis por Distancia a Centros de Empleo', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_dis.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)

    # Estadísticas por Número de Habitaciones
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '5. Análisis por Número de Habitaciones', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_rm.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)
    
    # Estadísticas por Nivel Socioeconómico
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '6. Análisis por Nivel Socioeconómico', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_lstat.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)
    
    # Estadísticas por Antigüedad
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '7. Análisis por Antigüedad de las Viviendas', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for idx, row in stats_by_age.iterrows():
        pdf.cell(0, 10, f'{idx}:', 0, 1, 'L')
        pdf.cell(0, 10, f'Precio Promedio: ${row["Precio Promedio"]:.2f}k', 0, 1, 'L')
    pdf.ln(5)
    
    # Resultados del Modelo
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '5. Resultados de los Modelos', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Modelo de Regresión:', 0, 1, 'L')
    pdf.cell(0, 10, f'Error Cuadrático Medio: {mse:.2f}', 0, 1, 'L')
    pdf.cell(0, 10, f'R² Score: {r2:.2f}', 0, 1, 'L')
    pdf.cell(0, 10, f'Precisión del Modelo de Clasificación: {accuracy:.2f}', 0, 1, 'L')
    
    # Variables más importantes
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '6. Variables más Importantes', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Para Regresión:', 0, 1, 'L')
    for idx, row in feature_importance_reg.head().iterrows():
        pdf.cell(0, 10, f'{row["Feature"]}: {row["Importance"]:.4f}', 0, 1, 'L')
    
    return pdf.output(dest='S').encode('latin-1')

    """
    Funcion para eliminar los outliers de las columnas especificadas usando el método
    de desviación estándar ya que el mismo es menos agresivo en la eliminacion de datos 
    y el dataset en estudio es pequeño
    
    
    Parámetros:
    df: DataFrame
    columns: lista de columnas para eliminar outliers
    n_std: número de desviaciones estándar para considerar un valor como outlier
    
    Retorna:
    DataFrame sin outliers
    """

def remove_outliers(df, columns, n_std=3):
   
    df_clean = df.copy()
    for column in columns:
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        
        # Calcular límites
        lower_bound = mean - (n_std * std)
        upper_bound = mean + (n_std * std)
        
        # Crear máscara para valores dentro de los límites
        mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean

    """
    Funcion principal 
    """

def main():
    st.title("Seleccione un dataset desde kaggle")
    
    # Input para el nombre del dataset
    dataset_name = st.text_input(
        "Introduce el nombre del dataset (formato: usuario/nombre-dataset)", value="altavish/boston-housing-dataset"
        
    )
    
    if st.button("Descargar y Cargar Dataset"):
        with st.spinner("Descargando dataset..."):
            if download_dataset(dataset_name):
                try:
                    # Buscar archivos CSV en el directorio data
                    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
                    if csv_files:
                        # Leer el primer archivo CSV encontrado
                        df = pd.read_csv(os.path.join('data', csv_files[0]))
                        st.success("Dataset cargado exitosamente!")
                        
                        # Mostrar información del dataset
                        st.write("Dimensiones del dataset:", df.shape)
                        st.subheader("Dataframe Original")
                        st.dataframe(df)
                        st.subheader("Informacion General")
                        info_df= pd.DataFrame({
                            "Columns": df.columns,
                            "Tipo de Dato": df.dtypes,
                            "Valores no Nulos":df.count(),
                            "Valores Nulos": df.isnull().sum()
                        })
                        st.dataframe(info_df)

                        st.subheader("Informacion Descriptiva")
                        st.write(df.describe())
                        
                        
                        #Esta parte del codigo es optima para cuando hay valores nulos en el dataset
                        
                        st.subheader("Dataframe sin valores nulos (df_clean)")
                        df_clean= df.fillna(df.mean(numeric_only=True)) #rellena con el promedio de los valores de la columna
                        st.write (df_clean)

                        st.write ("Resumen de busqueda de valores nulos en el df_clean")
                        nulos_summary = df_clean.isna().sum()
                        st.write (nulos_summary)
                        
                        # Análisis de correlación
                        st.subheader("Matriz de Correlación")
                        
                        # Obtener solo las columnas numéricas
                        numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
                        
                        
                        if len(numeric_columns) >0 :
                            # Calcular la matriz de correlación
                            correlation_matrix = df_clean[numeric_columns].corr()
                            
                            # Crear el mapa de calor con plotly
                            fig_corr = px.imshow(
                                correlation_matrix,
                                labels=dict(color="Correlación"),
                                x=correlation_matrix.columns,
                                y=correlation_matrix.columns,
                                color_continuous_scale='RdBu_r',  # Escala de colores rojo-azul
                                aspect='auto'
                            )
                            
                            # Personalizar el diseño
                            fig_corr.update_layout(
                                title="Mapa de Calor de Correlaciones",
                                xaxis_title="Variables",
                                yaxis_title="Variables",
                                width=800,
                                height=800
                            )
                            
                            # Mostrar el gráfico
                            st.plotly_chart(fig_corr, key= "Mapa de correlacion")

                            
                            #Anaisis de datos de los principales parametros del dataset
                            #de aqui en adelante funciona solo para el dataset de bienes raices de boston
                            #que se encuentra por defecto listo para descargar
                            
                            
                            # Análisis estadístico por rangos de criminalidad
                            st.subheader("Estadísticas de Precios por Nivel de Criminalidad")
                            
                            # Crear categorías de criminalidad
                            df_clean['CRIM_CAT'] = pd.qcut(df_clean['CRIM'], q=4, labels=[
                                'Muy Baja Criminalidad',
                                'Baja Criminalidad',
                                'Alta Criminalidad',
                                'Muy Alta Criminalidad'
                            ])

                            # Calcular estadísticas por categoría
                            stats_by_crim = df_clean.groupby('CRIM_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por nivel de criminalidad:")
                            st.dataframe(stats_by_crim)

                            # Crear un box plot para visualizar la distribución de precios por categoría
                            fig_box_crim = px.box(
                                df_clean,
                                x='CRIM_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Nivel de Criminalidad',
                                labels={
                                    'CRIM_CAT': 'Nivel de Criminalidad',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_crim.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_crim, key="box_crim")

                            # Análisis estadístico por rangos de PTRATIO (ratio alumno-profesor)
                            st.subheader("Estadísticas de Precios por Ratio Alumno-Profesor")
                            
                            # Crear categorías de PTRATIO
                            df_clean['PTRATIO_CAT'] = pd.qcut(df_clean['PTRATIO'], q=4, labels=[
                                'Ratio Muy Bajo',
                                'Ratio Bajo',

                                'Ratio Alto',
                                'Ratio Muy Alto'
                            ])

                            # Calcular estadísticas por categoría de PTRATIO
                            stats_by_ptratio = df_clean.groupby('PTRATIO_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por ratio alumno-profesor:")
                            st.dataframe(stats_by_ptratio)

                            # Box plot para PTRATIO
                            fig_box_ptratio = px.box(
                                df_clean,
                                x='PTRATIO_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Ratio Alumno-Profesor',
                                labels={
                                    'PTRATIO_CAT': 'Ratio Alumno-Profesor',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_ptratio.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_ptratio, key="box_ptratio")

                            # Análisis estadístico por rangos de DIS (distancia a centros de empleo)
                            st.subheader("Estadísticas de Precios por Distancia a Centros de Empleo")
                            
                            # Crear categorías de DIS
                            df_clean['DIS_CAT'] = pd.qcut(df_clean['DIS'], q=4, labels=[
                                'Muy Cerca',
                                'Cerca',
                                'Lejos',
                                'Muy Lejos'
                            ])

                            # Calcular estadísticas por categoría de DIS
                            stats_by_dis = df_clean.groupby('DIS_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por distancia a centros de empleo:")
                            st.dataframe(stats_by_dis)

                            # Box plot para DIS
                            fig_box_dis = px.box(
                                df_clean,
                                x='DIS_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Distancia a Centros de Empleo',
                                labels={
                                    'DIS_CAT': 'Distancia a Centros de Empleo',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_dis.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_dis, key="box_dis")

                            # Análisis estadístico por rangos de RM (número de habitaciones)
                            st.subheader("Estadísticas de Precios por Número de Habitaciones")

                            # Crear categorías de RM
                            df_clean['RM_CAT'] = pd.qcut(df_clean['RM'], q=4, labels=[
                                'Muy Pocas Habitaciones',
                                'Pocas Habitaciones',
                                'Muchas Habitaciones',
                                'Muy Muchas Habitaciones'
                            ])

                            # Calcular estadísticas por categoría de RM
                            stats_by_rm = df_clean.groupby('RM_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por número de habitaciones:")
                            st.dataframe(stats_by_rm)

                            # Box plot para RM
                            fig_box_rm = px.box(
                                df_clean,
                                x='RM_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Número de Habitaciones',
                                labels={
                                    'RM_CAT': 'Categoría de Habitaciones',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_rm.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_rm, key="box_rm")

                            # Análisis estadístico por rangos de LSTAT (% población de bajo estatus)
                            st.subheader("Estadísticas de Precios por Nivel Socioeconómico")

                            # Crear categorías de LSTAT
                            df_clean['LSTAT_CAT'] = pd.qcut(df_clean['LSTAT'], q=4, labels=[
                                'Nivel Socioeconómico Alto',
                                'Nivel Socioeconómico Medio-Alto',
                                'Nivel Socioeconómico Medio-Bajo',
                                'Nivel Socioeconómico Bajo'
                            ])

                            # Calcular estadísticas por categoría de LSTAT
                            stats_by_lstat = df_clean.groupby('LSTAT_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por nivel socioeconómico:")
                            st.dataframe(stats_by_lstat)

                            # Box plot para LSTAT
                            fig_box_lstat = px.box(
                                df_clean,
                                x='LSTAT_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Nivel Socioeconómico',
                                labels={
                                    'LSTAT_CAT': 'Nivel Socioeconómico',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_lstat.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_lstat, key="box_lstat")

                            # Análisis estadístico por rangos de AGE (antigüedad de las viviendas)
                            st.subheader("Estadísticas de Precios por Antigüedad de las Viviendas")

                            # Crear categorías de AGE
                            df_clean['AGE_CAT'] = pd.qcut(df_clean['AGE'], q=4, labels=[
                                'Construcción Reciente',
                                'Construcción Media',
                                'Construcción Antigua',
                                'Construcción Muy Antigua'
                            ])

                            # Calcular estadísticas por categoría de AGE
                            stats_by_age = df_clean.groupby('AGE_CAT')['MEDV'].agg([
                                ('Precio Promedio', 'mean'),
                                ('Precio Mínimo', 'min'),
                                ('Precio Máximo', 'max'),
                                ('Desviación Estándar', 'std')
                            ]).round(2)

                            st.write("Estadísticas de precios por antigüedad de las viviendas:")
                            st.dataframe(stats_by_age)

                            # Box plot para AGE
                            fig_box_age = px.box(
                                df_clean,
                                x='AGE_CAT',
                                y='MEDV',
                                title='Distribución de Precios por Antigüedad de las Viviendas',
                                labels={
                                    'AGE_CAT': 'Antigüedad de la Vivienda',
                                    'MEDV': 'Precio Medio de Vivienda (en $1000s)'
                                }
                            )

                            fig_box_age.update_layout(
                                height=500,
                                width=800
                            )

                            st.plotly_chart(fig_box_age, key="box_age")

                            
                            #Desde aqui se realiza el analisis de outliers, para visualizar si 
                            #estos vaores pueden afectar el modelo de ML
                            
                            st.subheader("Análisis de Outliers")

                            #Método del IQR (Rango Intercuartílico)
                            st.write("2. Detección de Outliers mediante el método IQR")
                            
                            for column in numeric_columns:
                                Q1 = df_clean[column].quantile(0.25)
                                Q3 = df_clean[column].quantile(0.75)
                                IQR = Q3 - Q1
                                
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                
                                outliers = df_clean[(df_clean[column] < lower_bound) | 
                                                  (df_clean[column] > upper_bound)][column]
                                
                                if len(outliers) > 0:
                                    st.write(f"\nOutliers encontrados en {column}:")
                                    st.write(f"- Límite inferior: {lower_bound:.2f}")
                                    st.write(f"- Límite superior: {upper_bound:.2f}")
                                    st.write(f"- Número de outliers: {len(outliers)}")
                                    
                                    # Se crea un Histograma con limites
                                    fig_hist = go.Figure()
                                    
                                    # se agrega el histograma aqui
                                    fig_hist.add_trace(go.Histogram(
                                        x=df_clean[column],
                                        name='Distribución',
                                        nbinsx=30
                                    ))
                                    
                                    # se agrega líneas verticales para los límites
                                    fig_hist.add_vline(x=lower_bound, 
                                                     line_dash="dash", 
                                                     line_color="red",
                                                     annotation_text="Límite inferior")
                                    fig_hist.add_vline(x=upper_bound, 
                                                     line_dash="dash", 
                                                     line_color="red",
                                                     annotation_text="Límite superior")
                                    
                                    fig_hist.update_layout(
                                        title=f"Distribución de {column} con límites de outliers",
                                        xaxis_title=column,
                                        yaxis_title="Frecuencia",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_hist)

                            
                           
                           #Construccion del modelo de ML con el modelo Randomforest
                            
                            st.header("Modelos de Machine Learning")
                            # Preparación de datos
                            features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

                            # Eliminacion outliers antes de crear los modelos
                            st.subheader("Eliminación de Outliers")
                            df_no_outliers = remove_outliers(df_clean, features + ['MEDV'])
                            st.write(f"Registros originales: {len(df_clean)}")
                            st.write(f"Registros después de eliminar outliers: {len(df_no_outliers)}")
                            st.write("Porcentaje de datos conservados: {:.2f}%".format(
                                 (len(df_no_outliers) / len(df_clean)) * 100
                                    ))

                            X = df_no_outliers[features]
                            y_reg = df_no_outliers['MEDV']  # Variable objetivo para regresión

                            # Creacion de la variable objetivo para clasificación (dividir precios en categorías)
                            y_class = pd.qcut(df_no_outliers['MEDV'], q=3, labels=['Bajo', 'Medio', 'Alto'])

                                # Division datos en conjunto de entrenamiento y prueba
                            X_train, X_test, y_reg_train, y_reg_test = train_test_split(
                                X, y_reg, test_size=0.2, random_state=42
                            )
                            _, _, y_class_train, y_class_test = train_test_split(
                                X, y_class, test_size=0.2, random_state=42
                            )

                            # Escalar características
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)

                            # Modelo de Regresión
                            st.subheader("Modelo de Regresión Random Forest")
                            
                            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf_reg.fit(X_train_scaled, y_reg_train)
                            
                            # Predicciones
                            y_reg_pred = rf_reg.predict(X_test_scaled)
                            
                            # Métricas de regresión
                            mse = mean_squared_error(y_reg_test, y_reg_pred)
                            r2 = r2_score(y_reg_test, y_reg_pred)
                            
                            st.write("Métricas del modelo de regresión:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Error Cuadrático Medio", f"{mse:.2f}")
                            with col2:
                                st.metric("R² Score", f"{r2:.2f}")

                            # Importancia de características para regresión
                            feature_importance_reg = pd.DataFrame({
                                'Feature': features,
                                'Importance': rf_reg.feature_importances_
                            }).sort_values('Importance', ascending=False)

                            st.write("Importancia de características (Regresión):")
                            fig_importance_reg = px.bar(
                                feature_importance_reg,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Importancia de Variables en la Predicción del Precio'
                            )
                            st.plotly_chart(fig_importance_reg)

                            # Modelo de Clasificación
                            st.subheader("Modelo de Clasificación Random Forest")
                            
                            rf_class = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf_class.fit(X_train_scaled, y_class_train)
                            
                            # Predicciones
                            y_class_pred = rf_class.predict(X_test_scaled)
                            
                            # Métricas de clasificación
                            accuracy = accuracy_score(y_class_test, y_class_pred)
                            
                            st.write("Métricas del modelo de clasificación:")
                            st.metric("Precisión (Accuracy)", f"{accuracy:.2f}")
                            
                            # Reporte de clasificación
                            st.write("Reporte detallado de clasificación:")
                            report = classification_report(y_class_test, y_class_pred)
                            st.text(report)

                            # Importancia de características para clasificación
                            feature_importance_class = pd.DataFrame({
                                'Feature': features,
                                'Importance': rf_class.feature_importances_
                            }).sort_values('Importance', ascending=False)

                            st.write("Importancia de características (Clasificación):")
                            fig_importance_class = px.bar(
                                feature_importance_class,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Importancia de Variables en la Clasificación del Precio'
                            )
                            st.plotly_chart(fig_importance_class)
                                                        
                            # Predictor con valores por default
                            st.subheader("Predictor con Valores por Default")
                            
                            # Definicion casos de ejemplo
                            example_cases = {
                                'Caso 1 (Zona Residencial)': {
                                    'CRIM': 0.1,      # Baja criminalidad
                                    'ZN': 18.0,       # Zona residencial
                                    'INDUS': 2.5,     # Baja industrialización
                                    'CHAS': 0,        # No junto al río
                                    'NOX': 0.4,       # Baja contaminación
                                    'RM': 6.5,        # Tamaño medio-alto
                                    'AGE': 45.0,      # Edad media
                                    'DIS': 4.5,       # Distancia media
                                    'RAD': 4,         # Accesibilidad media
                                    'TAX': 300,       # Impuestos medios
                                    'PTRATIO': 15,    # Ratio estudiante-profesor medio
                                    'B': 380,         # Población afroamericana media
                                    'LSTAT': 10.0     # Estatus socioeconómico medio
                                },
                                'Caso 2 (Zona Urbana)': {
                                    'CRIM': 0.5,
                                    'ZN': 0.0,
                                    'INDUS': 18.0,
                                    'CHAS': 1,
                                    'NOX': 0.6,
                                    'RM': 5.5,
                                    'AGE': 85.0,
                                    'DIS': 2.0,
                                    'RAD': 24,
                                    'TAX': 600,
                                    'PTRATIO': 20,
                                    'B': 380,
                                    'LSTAT': 20.0
                                },
                                'Caso 3 (Zona Suburbana)': {
                                    'CRIM': 0.02,
                                    'ZN': 80.0,
                                    'INDUS': 3.0,
                                    'CHAS': 0,
                                    'NOX': 0.3,
                                    'RM': 7.0,
                                    'AGE': 20.0,
                                    'DIS': 7.0,
                                    'RAD': 2,
                                    'TAX': 200,
                                    'PTRATIO': 13,
                                    'B': 380,
                                    'LSTAT': 5.0
                                }
                            }

                            # Se muestra la predicciones para cada caso
                            for case_name, case_values in example_cases.items():
                                st.write(f"\n### {case_name}")
                                
                                # Se muestra características del caso
                                st.write("Características:")
                                case_df = pd.DataFrame([case_values])
                                st.dataframe(case_df)
                                
                                # Preparacion datos y predicción
                                case_scaled = scaler.transform(case_df)
                                reg_pred = rf_reg.predict(case_scaled)[0]
                                class_pred = rf_class.predict(case_scaled)[0]
                                
                                # Se muestran las predicciones
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Precio Predicho", 
                                        f"${reg_pred:.2f}k",
                                        delta=f"{reg_pred - df_clean['MEDV'].mean():.2f}k vs precio medio"
                                    )
                                with col2:
                                    st.metric(
                                        "Categoría de Precio",
                                        class_pred
                                    )
                                
                                # Se Agrega una línea divisoria entre casos
                                st.markdown("---")

                            # se muestran estadísticas comparativas
                            st.subheader("Estadísticas Comparativas")
                            stats_df = pd.DataFrame({
                                'Métrica': ['Precio Medio Real', 'Precio Mínimo Real', 'Precio Máximo Real'],
                                'Valor': [
                                    f"${df_clean['MEDV'].mean():.2f}k",
                                    f"${df_clean['MEDV'].min():.2f}k",
                                    f"${df_clean['MEDV'].max():.2f}k"
                                ]
                            })
                            st.dataframe(stats_df)
                                                        
                            # Generación automática del informe
                            st.subheader("Informe del Análisis")
                            
                            # Generar el PDF
                            pdf_bytes = export_report(
                                df_clean,
                                stats_by_crim,
                                stats_by_ptratio,
                                stats_by_dis,
                                stats_by_rm,      
                                stats_by_lstat,   
                                stats_by_age,     
                                feature_importance_reg,
                                feature_importance_class,
                                mse,
                                r2,
                                accuracy
                            )
                            
                            # link de descarga
                            st.download_button(
                                label="📊 Descargar Informe Completo",
                                data=pdf_bytes,
                                file_name="informe_analisis_viviendas.pdf",
                                mime="application/pdf"
                            )
                            
                            # Mostrar mensaje informativo
                            st.info("""
                            💡 El informe incluye:
                            - Resumen del análisis de datos
                            - Estadísticas por nivel de criminalidad
                            - Análisis por ratio alumno-profesor
                            - Análisis por distancia a centros de empleo
                            - Resultados de los modelos predictivos
                            - Variables más importantes
                            """)
                        
                    else:
                        st.error("No se encontraron archivos CSV en el dataset descargado")
                except Exception as e:
                    st.error(f"Error al cargar el dataset: {str(e)}")

if __name__ == "__main__":
    main()