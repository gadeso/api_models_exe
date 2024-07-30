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
import git
import shutil

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Database configuration using environment variables
DATABASE_URL = {
    'host': os.getenv('DATABASE_HOST'),
    'user': os.getenv('DATABASE_USER'),
    'password': os.getenv('DATABASE_PASSWORD'),
    'db': os.getenv('DATABASE_DB')
}

# Load the model
model_path = 'models/final_model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)

# Function to get a database connection
def get_db_connection():
    return pymysql.connect(**DATABASE_URL)

@app.get("/predict")
async def predict(id_candidatura: int):
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Get candidate data
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

            # Create a DataFrame with candidate features and competencies
            input_data = pd.DataFrame([candidato_data], columns=[
                'edad', 'nota_media', 'nivel_ingles', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa'
            ])

            # Make the prediction
            prediction = model.predict(input_data)
            result = 'Admitido' if prediction == 1 else 'Rechazado'

            return {"id_candidatura": id_candidatura, "prediction": result}
    finally:
        connection.close()

@app.post("/retrain")
async def retrain():
    connection = get_db_connection()
    
    try:
        with connection.cursor() as cursor:
            # Get all competencies
            cursor.execute("""
                SELECT id_candidatura, nombre_competencia, nota
                FROM competencias
            """)
            competencias = cursor.fetchall()
            
            # Get all candidatures with the required status
            cursor.execute("""
                SELECT id_candidatura, status
                FROM candidaturas
                WHERE status IN ('Entrevista2', 'Ofertado', 'Entrevista1', 'CentroEvaluación', 'Descartado')
            """)
            candidaturas = cursor.fetchall()

            if not competencias or not candidaturas:
                raise HTTPException(status_code=404, detail="No se encontraron suficientes datos para reentrenar el modelo.")

            # Create a dictionary for competencies by candidature
            competencias_dict = {}
            for comp in competencias:
                if comp[0] not in competencias_dict:
                    competencias_dict[comp[0]] = {}
                competencias_dict[comp[0]][comp[1]] = comp[2]

            # Create lists for input data (X) and labels (y)
            X = []
            y = []

            required_competencies = ['Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']
            for cand in candidaturas:
                if cand[0] in competencias_dict:
                    comp_dict = competencias_dict[cand[0]]
                    # Ensure all required competencies are present
                    if all(comp in comp_dict for comp in required_competencies):
                        # Get candidate data
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

            # Convert to DataFrame
            X_df = pd.DataFrame(X, columns=['edad', 'nota_media', 'nivel_ingles', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa'])
            y_series = pd.Series(y)

            # Define categorical and numerical columns
            categorical_features = ['nivel_ingles']
            numerical_features = ['edad', 'nota_media', 'Profesionalidad', 'Dominio', 'Resiliencia', 'HabilidadesSociales', 'Liderazgo', 'Colaboracion', 'Compromiso', 'Iniciativa']

            # Preprocessing for categorical and numerical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])

            # Define the pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])

            # Train the model
            new_model = pipeline.fit(X_df, y_series)

            # Save the new model
            with open(model_path, 'wb') as f:
                joblib.dump(new_model, f)

            # Git operations to update the model in the repository
            repo_url = 'https://github.com/sebasg2/api_modelo_exe.git'
            repo_path = '/tmp/api_modelo_exe'  # Temporary local path
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)  # Remove existing temp directory to avoid conflicts
            repo = git.Repo.clone_from(repo_url, repo_path)
            model_repo_path = os.path.join(repo_path, model_path)
            shutil.copy2(model_path, model_repo_path)  # Copy the updated model to the cloned repo path
            repo.index.add([model_repo_path])
            repo.index.commit('Updated model after retraining')
            origin = repo.remote(name='origin')
            origin.push()

            return {"message": "Modelo reentrenado y actualizado en el repositorio exitosamente."}
    finally:
        connection.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_modelo:app", host="0.0.0.0", port=8000)

