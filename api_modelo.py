from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pymysql
import pandas as pd
from typing import Optional
import joblib
from sklearn.ensemble import RandomForestClassifier


# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las credenciales desde las variables de entorno
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
database = os.getenv('DB_DATABASE')

config = {
    'user': username,
    'password': password,
    'host': host,
    'port': 3306,  # Asegúrate de incluir el puerto si es necesario
    'database': database,
    'cursorclass': pymysql.cursors.DictCursor
}

app = FastAPI()

# Configuración del middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow only the specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Load the model
model_path = '../models/modelo_seleccion.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
else:
    raise FileNotFoundError("El archivo del modelo no se encontró.")

# Function to get a database connection
def get_db_connection():
    return pymysql.connect(**config)

@app.get("/predict")
def predict(id_candidatura: int):
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Obtener las notas de competencias de la candidatura
            cursor.execute("SELECT nombre_competencia, nota FROM competencias WHERE id_candidatura = %s", (id_candidatura,))
            competencias = cursor.fetchall()

            if not competencias:
                raise HTTPException(status_code=404, detail="No se encontraron competencias para la candidatura proporcionada.")
            
            # Crear un diccionario para las competencias
            competencias_dict = {comp['nombre_competencia']: comp['nota'] for comp in competencias}

            # Verificar que todas las competencias necesarias están presentes
            required_competencies = ['Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']
            for comp in required_competencies:
                if comp not in competencias_dict:
                    raise HTTPException(status_code=400, detail=f"La competencia {comp} no está presente en la candidatura.")

            # Crear un DataFrame con las competencias en el orden correcto
            input_data = pd.DataFrame([[
                competencias_dict['Profesionalidad'],
                competencias_dict['Dominio'],
                competencias_dict['Resiliencia'],
                competencias_dict['HabilidadesSociales'],
                competencias_dict['Liderazgo'],
                competencias_dict['Colaboracion'],
                competencias_dict['Compromiso'],
                competencias_dict['Iniciativa']
            ]])

            # Realizar la predicción
            prediction = model.predict(input_data)
            result = 'Admitido' if prediction == 1 else 'Rechazado'

            return {"id_candidatura": id_candidatura, "prediction": result}
    finally:
        connection.close()


@app.post("/retrain")
def retrain():
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Obtener todas las competencias
            cursor.execute("SELECT id_candidatura, nombre_competencia, nota FROM competencias")
            competencias = cursor.fetchall()
            
            # Obtener todas las candidaturas con el status requerido
            cursor.execute("SELECT id_candidatura, status FROM candidaturas WHERE status IN ('Entrevista2', 'Ofertado', 'Entrevista1', 'CentroEvaluación', 'Descartado')")
            candidaturas = cursor.fetchall()

            if not competencias or not candidaturas:
                raise HTTPException(status_code=404, detail="No se encontraron suficientes datos para reentrenar el modelo.")

            # Crear un diccionario para las competencias por candidatura
            competencias_dict = {}
            for comp in competencias:
                if comp['id_candidatura'] not in competencias_dict:
                    competencias_dict[comp['id_candidatura']] = {}
                competencias_dict[comp['id_candidatura']][comp['nombre_competencia']] = comp['nota']

            # Crear listas para los datos de entrada (X) y las etiquetas (y)
            X = []
            y = []

            required_competencies = ['Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']
            for cand in candidaturas:
                if cand['id_candidatura'] in competencias_dict:
                    comp_dict = competencias_dict[cand['id_candidatura']]
                    # Verificar que todas las competencias necesarias están presentes
                    if all(comp in comp_dict for comp in required_competencies):
                        X.append([comp_dict[comp] for comp in required_competencies])
                        y.append(1 if cand['status'] in ["Entrevista2", "Ofertado", "Entrevista1", "CentroEvaluación"] else 0)

            if not X or not y:
                raise HTTPException(status_code=400, detail="No se encontraron suficientes datos válidos para reentrenar el modelo.")

            # Convertir a DataFrame
            X = pd.DataFrame(X, columns=required_competencies)
            y = pd.Series(y)

            # Entrenar el modelo
            new_model = RandomForestClassifier()
            new_model.fit(X, y)

            # Guardar el nuevo modelo
            joblib.dump(new_model, model_path)

            return {"detail": "Modelo reentrenado con éxito"}
    finally:
        connection.close()
