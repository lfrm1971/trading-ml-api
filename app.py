from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# Cargar modelo e información
print("📦 Cargando modelo XGBoost...")
with open('modelo_trading.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('model_info.json', 'r', encoding='utf-8') as f:
    model_info = json.load(f)

print("✅ Modelo cargado correctamente")
print(f"   Tipo: {model_info['model_type']}")
print(f"   Features: {len(model_info['features'])}")
print(f"   Clases: {model_info['classes']}")

@app.route('/')
def home():
    return """
    <h1>🤖 API de Predicción - Modelo de Trading XGBoost</h1>
    <p>Servidor activo y funcionando</p>
    <h3>Endpoints disponibles:</h3>
    <ul>
        <li><b>GET /info</b> - Información del modelo</li>
        <li><b>POST /predecir</b> - Hacer predicción</li>
    </ul>
    """

@app.route('/info', methods=['GET'])
def get_info():
    return jsonify(model_info)

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        datos = request.json
        
        if 'features' not in datos:
            return jsonify({'error': 'Falta el campo "features"'}), 400
        
        features = datos['features']
        
        if len(features) != len(model_info['features']):
            return jsonify({
                'error': f'Se esperan {len(model_info["features"])} features, se recibieron {len(features)}'
            }), 400
        
        X = np.array([features])
        prediccion_numerica = int(modelo.predict(X)[0])
        prediccion_texto = model_info['classes'][prediccion_numerica]
        
        try:
            probabilidades = modelo.predict_proba(X)[0].tolist()
            proba_dict = {
                model_info['classes'][i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilidades)
            }
        except:
            proba_dict = None
        
        respuesta = {
            'prediccion': prediccion_texto,
            'prediccion_numerica': prediccion_numerica,
            'clase_color': model_info['class_colors'][str(prediccion_numerica)],
            'probabilidades': proba_dict,
            'features_recibidas': dict(zip(model_info['features'], features))
        }
        
        return jsonify(respuesta)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)