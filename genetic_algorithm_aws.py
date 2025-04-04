# Importing dependencies for DEAP
from Managers.GameDirector import GameDirector
from Agents.TristanAgent import TristanAgent as ta
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.EdoAgent import EdoAgent as ea
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.RandomAgent import RandomAgent as ra
import time
import requests
import random
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from deap import base, creator, tools

# -------------------------------------------------------------------
# 0. Definicion de maquinas EC2
# -------------------------------------------------------------------
EC2_WORKERS = [
    "http://3.227.2.78:8000/evaluate",
    "http://3.215.177.175:8000/evaluate",
    "http://3.235.31.68:8000/evaluate",
    "http://13.217.208.142:8000/evaluate",
    "http://3.236.66.160:8000/evaluate",
    "http://18.208.143.199:8000/evaluate",
    "http://100.27.251.134:8000/evaluate",
    "http://13.216.2.22:8000/evaluate"
]


def distributed_http_map(individuals):
    n_workers = len(EC2_WORKERS)
    # divide de forma más pareja
    batches = np.array_split(individuals, n_workers)

    results = []

    def send_request(url, batch):
        try:
            json_data = {"individuals": [list(ind) for ind in batch]}
            response = requests.post(url, json=json_data)
            response.raise_for_status()
            fitnesses = response.json()["fitnesses"]
            return [(f,) for f in fitnesses]
        except Exception as e:
            print(f"Error en {url}: {e}")
            return [(0.0,) for _ in batch]

    urls = cycle(EC2_WORKERS)  # En caso de que batches > workers

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(send_request, next(urls), batch)
                   for batch in batches]
        for future in futures:
            results.extend(future.result())

    return results


# -------------------------------------------------------------------
# 1. Definir Estructuras de Fitness e Individuo
# -------------------------------------------------------------------
# Suponiendo que queremos maximizar la función de evaluación:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# -------------------------------------------------------------------
# 2. Inicialización del Individuo (lista de 10 probabilidades que suman 1)
# -------------------------------------------------------------------


def initIndividual(icls, size):
    # Genera 10 números aleatorios y normaliza para que sumen 1
    vec = [random.random() for _ in range(size)]
    s = sum(vec)
    individual = [x / s for x in vec]
    return icls(individual)


toolbox = base.Toolbox()
toolbox.register("individual", initIndividual, creator.Individual, size=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Configuración de paralelización
toolbox.register("individual", initIndividual, creator.Individual, size=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -------------------------------------------------------------------
# 3. Función de Simulación (Ejemplo ilustrativo)
# -------------------------------------------------------------------

AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]


def simular_catan(idx_agents_to_play: list):
    try:
        agents_to_play = [AGENTS[idx] for idx in idx_agents_to_play]
        game_director = GameDirector(
            agents=agents_to_play, max_rounds=200, store_trace=False)

        game_trace = game_director.game_start(print_outcome=False)
    except Exception as e:
        print(f"Error: {e}")
        raise e

    # Análisis de resultados
    last_round = max(game_trace["game"].keys(),
                     key=lambda r: int(r.split("_")[-1]))
    last_turn = max(game_trace["game"][last_round].keys(
    ), key=lambda t: int(t.split("_")[-1].lstrip("P")))

    victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

    return victory_points

# -------------------------------------------------------------------
# 4. Función de Evaluación
# -------------------------------------------------------------------


def evaluate(individual):
    num_partidas = 50
    total_score = 0.0
    results = []
    for _ in range(num_partidas):
        # 1. Seleccionar el agente evaluado usando las probabilidades del individuo
        idx_selected_agent = int(np.random.choice(range(10), p=individual))

        # 2. Seleccionar 3 agentes adicionales de entre los 9 restantes
        players = [i for i in range(10) if i != idx_selected_agent]
        players = random.sample(players, 3)

        # 3. Ubicar en una posicion aleatoria al jugador seleccionado
        idx_player_in_game = random.randint(0, 3)
        players = players[:idx_player_in_game] + \
            [idx_selected_agent] + players[idx_player_in_game:]

        try:
            results = simular_catan(players)

            # Convertir a lista ordenada por puntaje obtenido
            sorted_indices = sorted(results.keys(),
                                    key=lambda k: int(results[k]))
            ordered_idx = [int(k[1:]) for k in sorted_indices]

            # Se calcula el score considerando posicion en los resultados finales
            # El maximo puntaje es 1 si tiene mayor cantidad de puntos de victoria
            # Se divide entre 3 porque el mayor indice es 3.
            total_score += ordered_idx.index(idx_player_in_game) / 3
        except Exception as ex:
            total_score += 0

    return total_score / num_partidas,


toolbox.register("evaluate", evaluate)

# -------------------------------------------------------------------
# 5. Registro de Operadores Genéticos
# -------------------------------------------------------------------

# Selección por torneo
toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("select", tools.selRoulette)

# Cruce: blend crossover con normalización


def cxBlendNormalize(ind1, ind2, alpha=0.8):
    """
    Se utilizará cxBlend, la cual realiza lo siguiente:
    1. Se halla una diferencia:
      d = |gen_1 - gen_2|
    2. Se calculan los limites:
      Limite inferior: min(gen_1, gen_2) - alpha x d
      Limite superior: max(gen_1, gen_2) + alpha x d
    3. Generación de nuevo gen
      nuevo_gen = random.uniform(Limite_inferior, Limite_superior)
    """
    tools.cxBlend(ind1, ind2, alpha)
    # Asegurarse de que ningún valor sea negativo
    ind1[:] = [max(x, 0.001) for x in ind1]
    ind2[:] = [max(x, 0.001) for x in ind2]
    s1, s2 = sum(ind1), sum(ind2)
    ind1[:] = [x / s1 for x in ind1]
    ind2[:] = [x / s2 for x in ind2]
    return ind1, ind2


toolbox.register("mate", cxBlendNormalize)

# Mutación: mutación gaussiana con normalización


def mutGaussianNormalize(individual, mu, sigma, indpb):
    tools.mutGaussian(individual, mu, sigma, indpb)
    individual[:] = [max(x, 0.001) for x in individual]
    total = sum(individual)
    if total != 0:
        individual[:] = [x / total for x in individual]
    return individual,


toolbox.register("mutate", mutGaussianNormalize, mu=0, sigma=0.1, indpb=0.2)

# -------------------------------------------------------------------
# 6. Ciclo Evolutivo
# -------------------------------------------------------------------


def main():
    start_time = time.time()  # Inicio del contador
    random.seed(42)
    # Población de 20 individuos
    pop = toolbox.population(n=70)

    # Parámetros del algoritmo evolutivo:
    CXPB = 0.8   # Probabilidad de cruzamiento
    MUTPB = 0.2  # Probabilidad de mutación
    NGEN = 150    # Número de generaciones

    best_fitness_global = -float("inf")  # Inicia con el peor valor posible
    best_individual_global = None  # Para almacenar el mejor individuo

    print("Inicio de la evolución")

    # Evaluación inicial de toda la población
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("Evaluación inicial completada")

    # Listas para almacenar el fitness máximo y medio de cada generación
    max_fitness_values = []
    avg_fitness_values = []

    # Bucle evolutivo
    for gen in range(NGEN):
        print(f"--- Generación {gen} ---")
        # Seleccionar la descendencia
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Aplicar cruce en parejas
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Aplicar mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Reevaluar a los individuos que se han modificado
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Actualizar la población
        pop[:] = offspring

        # Estadísticas de la generación actual
        fits = [ind.fitness.values[0] for ind in pop]
        max_fit = max(fits)
        avg_fit = sum(fits) / len(fits)
        max_fitness_values.append(max_fit)
        avg_fitness_values.append(avg_fit)

        print(f"Máximo fitness: {max_fit}")
        print(f"Fitness promedio: {avg_fit}\n")

        # Obtener el mejor individuo de esta generación
        best_individual = tools.selBest(pop, 1)[0]
        best_fitness_current = best_individual.fitness.values[0]

        # Comparar con el mejor fitness global y actualizar si es necesario
        if best_fitness_current > best_fitness_global:
            best_fitness_global = best_fitness_current
            best_individual_global = best_individual

    end_time = time.time()  # Fin del contador
    elapsed_time = end_time - start_time
    print(f"\n⏱ Tiempo total de ejecución: {elapsed_time:.2f} segundos")
    print(f"\nMejor fitness global alcanzado: {best_fitness_global}")
    print(f"Mejor individuo encontrado: {best_individual_global}")

    # Graficar la evolución del fitness
    step = max(1, NGEN // 20)
    plt.figure()
    plt.plot(range(NGEN), avg_fitness_values, label="Fitness Medio")
    plt.plot(range(NGEN), max_fitness_values, label="Fitness Máximo")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness")
    # Asegura que los ticks sean enteros
    plt.xticks(np.arange(0, NGEN, step=step), rotation=45)
    plt.legend()
    plt.savefig("evolucion_fitness.png")


if __name__ == "__main__":
    # toolbox.register("map", distributed_http_map)
    toolbox.register("map", lambda f, iterable: distributed_http_map(iterable))
    main()
