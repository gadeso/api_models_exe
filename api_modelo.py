from fastapi import FastAPI, HTTPException
import pymysql
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

app = FastAPI()

# Configuración de la base de datos usando las variables de entorno
DATABASE_URL = {
    'host': os.getenv('DATABASE_HOST'),
    'user': os.getenv('DATABASE_USER'),
    'password': os.getenv('DATABASE_PASSWORD'),
    'db': os.getenv('DATABASE_DB')
}

# Cargar el modelo
model_path = 'models/final_model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)

# Función para obtener una conexión a la base de datos
def get_db_connection():
    return pymysql.connect(**DATABASE_URL)

@app.get("/predict")
async def predict(id_candidatura: int):
    # Conectar a la base de datos
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Obtener los datos del candidato
            cursor.execute("""
                SELECT c.edad, c.nota_media, c.nivel_ingles, 
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Profesionalidad'), 0) AS Profesionalidad,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Dominio'), 0) AS Dominio,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Resiliencia'), 0) AS Resiliencia,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'HabilidadesSociales'), 0) AS HabilidadesSociales,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Liderazgo'), 0) AS Liderazgo,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Colaboracion'), 0) AS Colaboracion,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Compromiso'), 0) AS Compromiso,
                       IFNULL((SELECT nota FROM competencias WHERE id_candidatura = %s AND nombre_competencia = 'Iniciativa'), 0) AS Iniciativa
                FROM candidatos c
                INNER JOIN candidaturas cand ON c.id_candidato = cand.id_candidato
                WHERE cand.id_candidatura = %s
            """, (id_candidatura, id_candidatura, id_candidatura, id_candidatura, id_candidatura, id_candidatura, id_candidatura, id_candidatura, id_candidatura))
            candidato_data = cursor.fetchone()
            
            if not candidato_data:
                raise HTTPException(status_code=404, detail="No se encontraron datos para la candidatura proporcionada.")

            # Crear un DataFrame con las características del candidato y las competencias en el orden correcto
            input_data = pd.DataFrame([candidato_data], columns=[
                'edad', 'nota_media', 'nivel_ingles', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa'
            ])

            # Realizar la predicción
            prediction = model.predict(input_data)
            result = 'Admitido' if prediction == 1 else 'Rechazado'

            return {"id_candidatura": id_candidatura, "prediction": result}
    finally:
        connection.close()

@app.post("/retrain")
async def retrain():
    # Conectar a la base de datos
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Obtener todas las competencias
            cursor.execute("""
                SELECT id_candidatura, nombre_competencia, nota
                FROM competencias
            """)
            competencias = cursor.fetchall()
            
            # Obtener todas las candidaturas con el status requerido
            cursor.execute("""
                SELECT id_candidatura, status
                FROM candidaturas
                WHERE status IN ('Entrevista2', 'Ofertado', 'Entrevista1', 'CentroEvaluación', 'Descartado')
            """)
            candidaturas = cursor.fetchall()

            if not competencias or not candidaturas:
                raise HTTPException(status_code=404, detail="No se encontraron suficientes datos para reentrenar el modelo.")

            # Crear un diccionario para las competencias por candidatura
            competencias_dict = {}
            for comp in competencias:
                if comp[0] not in competencias_dict:
                    competencias_dict[comp[0]] = {}
                competencias_dict[comp[0]][comp[1]] = comp[2]

            # Crear listas para los datos de entrada (X) y las etiquetas (y)
            X = []
            y = []

            required_competencies = ['Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']
            for cand in candidaturas:
                if cand[0] in competencias_dict:
                    comp_dict = competencias_dict[cand[0]]
                    # Verificar que todas las competencias necesarias están presentes
                    if all(comp in comp_dict for comp in required_competencies):
                        # Obtener los datos del candidato
                        cursor.execute("""
                            SELECT c.edad, c.nota_media, c.nivel_ingles
                            FROM candidatos c
                            INNER JOIN candidaturas cand ON c.id_candidato = cand.id_candidato
                            WHERE cand.id_candidatura = %s
                        """, (cand[0],))
                        candidato_data = cursor.fetchone()
                        
                        if candidato_data:
                            X.append([
                                candidato_data[0],  # edad
                                candidato_data[1],  # nota_media
                                candidato_data[2],  # nivel_ingles
                                comp_dict.get('Profesionalidad', 0),
                                comp_dict.get('Dominio', 0),
                                comp_dict.get('Resiliencia', 0),
                                comp_dict.get('HabilidadesSociales', 0),
                                comp_dict.get('Liderazgo', 0),
                                comp_dict.get('Colaboracion', 0),
                                comp_dict.get('Compromiso', 0),
                                comp_dict.get('Iniciativa', 0)
                            ])
                            y.append(1 if cand[1] in ["Entrevista2", "Ofertado", "Entrevista1", "CentroEvaluación"] else 0)

            if not X or not y:
                raise HTTPException(status_code=400, detail="No se encontraron suficientes datos válidos para reentrenar el modelo.")

            # Convertir a DataFrame
            X_df = pd.DataFrame(X, columns=['edad', 'nota_media', 'nivel_ingles', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa'])
            y_series = pd.Series(y)

            # Definir las columnas categóricas y numéricas
            categorical_features = ['nivel_ingles']
            numerical_features = ['edad', 'nota_media', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']

            # Preprocesamiento para las características categóricas y numéricas
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])

            # Definir el pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            # Entrenar el modelo
            new_model = pipeline.fit(X_df, y_series)

            # Guardar el nuevo modelo
            with open(model_path, 'wb') as f:
                joblib.dump(new_model, f)

            return {"message": "Modelo reentrenado exitosamente."}
    finally:
        connection.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
