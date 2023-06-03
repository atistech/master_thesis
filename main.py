from flask import Flask, render_template, request, jsonify, redirect, session
from GeneticAlgorithm import GeneticAlgorithm

app = Flask(__name__)
ga = object


@app.route("/")
def index():
    return render_template("main.html")

@app.route("/ga-results/<gen>")
def ga_results(gen):
    global ga
    ga = GeneticAlgorithm(param_dict={
        "datasetSelection": "Mnist",
        "populationSize": 2
    })

    ga.initialPopulation()
    results = ga.calculateFitness()
    return jsonify(data=[r.serialize() for r in results])

@app.route("/ga-calback/<gen>")
def ga_callback(gen):
    global ga
    ga.selection()
    ga.crossOver()
    ga.mutation()
    results = ga.calculateFitness()
    return jsonify(data=[r.serialize() for r in results])

if __name__ == "__main__":
    app.run(debug=True)