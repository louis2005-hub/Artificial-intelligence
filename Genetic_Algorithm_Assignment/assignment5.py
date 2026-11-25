import random

# Distance matrix (in kilometres) for locations A through G (symmetric)
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DEPOT = "A"
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]

DISTANCE_MATRIX = {
    "A": {"A": 0, "B": 4, "C": 8, "D": 6, "E": 7, "F": 9, "G": 5},
    "B": {"A": 4, "B": 0, "C": 5, "D": 7, "E": 3, "F": 8, "G": 6},
    "C": {"A": 8, "B": 5, "C": 0, "D": 4, "E": 6, "F": 3, "G": 7},
    "D": {"A": 6, "B": 7, "C": 4, "D": 0, "E": 5, "F": 6, "G": 8},
    "E": {"A": 7, "B": 3, "C": 6, "D": 5, "E": 0, "F": 4, "G": 6},
    "F": {"A": 9, "B": 8, "C": 3, "D": 6, "E": 4, "F": 0, "G": 5},
    "G": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 6, "F": 5, "G": 0},
}


def calculate_route_distance(chromosome):
    """Calculate total round-trip distance starting and ending at DEPOT ('A')."""
    full_route = [DEPOT] + chromosome + [DEPOT]
    total_distance = 0
    for i in range(len(full_route) - 1):
        origin = full_route[i]
        destination = full_route[i + 1]
        total_distance += DISTANCE_MATRIX[origin][destination]
    return total_distance


def fitness_score(chromosome):
    """Higher fitness corresponds to a shorter total route distance."""
    return 1.0 / calculate_route_distance(chromosome)


def create_initial_population(pop_size):
    """Generate an initial population of valid route permutations (stops B-G)."""
    population = []
    for _ in range(pop_size):
        route = DELIVERY_STOPS.copy()
        random.shuffle(route)
        population.append(route)
    return population


def tournament_selection(population, tournament_size=3):
    """Select one parent route using tournament selection based on fitness."""
    competitors = random.sample(population, tournament_size)
    competitors.sort(key=fitness_score, reverse=True)
    return competitors[0].copy()


def order_crossover(parent1, parent2):
    """Perform Order Crossover (OX) to create a valid child without duplicates."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    # Copy slice from parent1
    child[start : end + 1] = parent1[start : end + 1]

    # Fill remaining positions from parent2 in order
    placed = set(child[start : end + 1])
    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_index] in placed:
                p2_index += 1
            child[i] = parent2[p2_index]
            p2_index += 1

    return child


def swap_mutation(chromosome, mutation_rate=0.15):
    """Swap two delivery stops with probability mutation_rate."""
    mutated = chromosome.copy()
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated


def run_genetic_algorithm(
    pop_size=60,
    generations=100,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elitism_count=2,
    seed=42,
):
    """Run the GA for delivery route optimization and return best route & distance."""
    random.seed(seed)
    population = create_initial_population(pop_size)

    for _ in range(generations):
        # Rank population from shortest distance (best) to longest
        population.sort(key=calculate_route_distance)
        next_population = [route.copy() for route in population[:elitism_count]]

        while len(next_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = swap_mutation(child, mutation_rate=mutation_rate)
            next_population.append(child)

        population = next_population

    population.sort(key=calculate_route_distance)
    best_chromosome = population[0]
    best_distance = calculate_route_distance(best_chromosome)
    best_full_route = [DEPOT] + best_chromosome + [DEPOT]
    return best_full_route, best_distance


if __name__ == "__main__":
    best_route, best_km = run_genetic_algorithm(
        pop_size=60,
        generations=100,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_count=2,
        seed=42,
    )
    route_str = " -> ".join(best_route)
    print(f"Best Route: {route_str}")
    print(f"Total Distance: {best_km} km")