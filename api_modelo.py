from fastapi import FastAPI, HTTPException
import pymysql
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

app = FastAPI()

# Configuración de la base de datos usando las variables de entorno
config = {
    'user': os.getenv('DATABASE_USER'),
    'password': os.getenv('DATABASE_PASSWORD'),
    'host': os.getenv('DATABASE_HOST'),
    'port': 3306,  # Asegúrate de incluir el puerto si es necesario
    'database': os.getenv('DATABASE_DB'),
    'cursorclass': pymysql.cursors.DictCursor
}

# Cargar el modelo
model_path = 'models/final_model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)

@app.get("/predict")
async def predict(id_candidatura: int):
    # Conectar a la base de datos
    db = pymysql.connect(**config)
    cursor = db.cursor()

    try:
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
        cursor.close()
        db.close()
