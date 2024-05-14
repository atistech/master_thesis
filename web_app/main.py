from flask import Flask, render_template, request, jsonify, redirect, session
import os, sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)
from src.genetic_algorithm import GeneticAlgorithm

app = Flask(__name__)
ga = object


@app.route("/")
def index():
    return render_template("main.html")

@app.route("/ga-results/<option>")
def ga_results(option):
    global ga
    ga = GeneticAlgorithm(param_dict={
        "dataset": request.files.get('dataset'),
        "populationSize": 2,
        "IsRegression": True if (option == "true") else False
    })

    results = ga.initialPopulation()
    return jsonify(data=[r.serialize() for r in results])

@app.route("/ga-calback")
def ga_callback():
    global ga
    results = ga.callback()
    return jsonify(data=[r.serialize() for r in results])

if __name__ == "__main__":
    app.run(debug=True)