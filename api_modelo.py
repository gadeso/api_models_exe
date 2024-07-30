from fastapi import FastAPI, HTTPException
import pymysql
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

app = FastAPI()

# Configuración de la base de datos
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

# Cargar el modelo
model_path = 'models/model_web.pkl'  # Actualiza el nombre del archivo del modelo
with open(model_path, 'rb') as f:
    model = joblib.load(f)

@app.get("/predict")
async def predict(id_candidatura: int):
    # Conectar a la base de datos
    db = pymysql.connect(**config)
    cursor = db.cursor()
    
    try:
        # Obtener las notas de competencias de la candidatura
        cursor.execute("""
            SELECT nombre_competencia, nota
            FROM competencias
            WHERE id_candidatura = %s
        """, (id_candidatura,))
        
        competencias = cursor.fetchall()

        if not competencias:
            raise HTTPException(status_code=404, detail="No se encontraron competencias para la candidatura proporcionada.")

        # Crear un diccionario para las competencias
        competencias_dict = {comp['nombre_competencia']: comp['nota'] for comp in competencias}

        # Verificar que todas las competencias necesarias están presentes
        required_competencies = ['Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']
        for comp in required_competencies:
            if comp not in competencias_dict:
                competencias_dict[comp] = 0  # Asignar 0 si la competencia no está presente

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
        ]], columns=[
            'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa'
        ])

        # Realizar la predicción
        prediction = model.predict(input_data)
        result = 'Admitido' if prediction == 1 else 'Rechazado'

        return {"prediction": result}

    finally:
        cursor.close()
        db.close()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_modelo:app", host="0.0.0.0", port=8000)