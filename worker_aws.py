from flask import Flask, request, jsonify
from genetic_algorithm import evaluate

app = Flask(__name__)


@app.route('/evaluate', methods=['POST'])
def evaluate_batch():
    data = request.get_json()
    individuals = data.get("individuals", [])

    results = []
    for ind in individuals:
        try:
            fitness = evaluate(ind)[0]
        except Exception as e:
            print(f"Error evaluando individuo: {e}")
            fitness = 0.0  # Penaliza en caso de error
        results.append(fitness)

    return jsonify({"fitnesses": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
