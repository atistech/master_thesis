from flask import Flask, render_template, request, jsonify, redirect, session
from GeneticAlgorithm import GeneticAlgorithm

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("main.html")

@app.route("/ga-results")
def ga_results():
    generationCount = 0
    ga = GeneticAlgorithm(param_dict={
        "datasetSelection": "Mnist",
        "populationSize": 2
    })

    ga.initialPopulation()
    results = ga.calculateFitness()
    return jsonify(data=[r.serialize() for r in results], gen=generationCount)


if __name__ == "__main__":
    app.run(debug=True)