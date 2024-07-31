import numpy as np


def initialize_environment(observation_tensor, pred_agent_position):
    """
    Initialize the environment based on the observation tensor.
    
    Args:
    observation_tensor (numpy.array): A NxNx3 tensor representing the partial observation.
                                      - Channel 1: Obstacles
                                      - Channel 2: Free space
                                      - Channel 3: Goal

    Returns:
    dict: A dictionary containing the initialized map, agent position, and goal position.
    """
    # Extract channels from the observation tensor
    obstacle_channel = observation_tensor[:, :, 0]
    free_space_channel = observation_tensor[:, :, 1]
    goal_channel = observation_tensor[:, :, 2]

    # Initialize map (0: unknown, 1: free space, -1: obstacle)
    env_map = np.zeros_like(obstacle_channel)
    env_map[free_space_channel == 1] = 1
    env_map[obstacle_channel == 1] = -1

    # Find the goal position (if visible)
    goal_position = None
    goal_indices = np.argwhere(goal_channel == 1)
    if goal_indices.size > 0:
        goal_position = goal_indices[0]

    return {
        "map": env_map,
        "agent_position": pred_agent_position,
        "goal_position": goal_position
    }

def identify_frontiers(env_map):
    """
    Identify frontiers in the environment map.

    Args:
    env_map (numpy.array): The map of the environment where 0 represents unknown, 1 is free space, and -1 is an obstacle.
    agent_position (numpy.array): The current position of the agent.

    Returns:
    list of tuples: A list of coordinates representing the frontiers.
    """
    frontiers = []
    rows, cols = env_map.shape

    # Directions to look for frontiers (up, down, left, right)
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    # Iterate over each cell in the map
    for row in range(rows):
        for col in range(cols):
            # Check if the cell is a free space
            if env_map[row, col] == 1:
                print(f"({row}, {col})")
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    # Check if the adjacent cell is unknown and within the map boundaries
                    if 0 <= new_row < rows and 0 <= new_col < cols and env_map[new_row, new_col] == 0:
                        frontiers.append((row, col))
                        break

    return frontiers

import heapq

def find_nearest_frontier(frontiers, agent_position):
    """
    Find the nearest frontier to the agent.

    Args:
    frontiers (list of tuples): A list of coordinates representing the frontiers.
    agent_position (numpy.array): The current position of the agent.

    Returns:
    tuple: The coordinates of the nearest frontier.
    """
    # Use a min heap to find the nearest frontier based on Euclidean distance
    frontier_distances = []
    for frontier in frontiers:
        distance = np.linalg.norm(np.array(frontier) - agent_position)
        heapq.heappush(frontier_distances, (distance, frontier))

    # Return the nearest frontier
    return heapq.heappop(frontier_distances)[1]


def plan_path_to_frontier(env_map, agent_position, target_frontier, action_set):
    """
    Plan a path from the agent's position to the target frontier.

    Args:
    env_map (numpy.array): The map of the environment.
    agent_position (numpy.array): The current position of the agent.
    target_frontier (tuple): The target frontier coordinates.

    Returns:
    list of tuples: The path from the agent to the frontier.
    """
    # Find the path to the target frontier using A* search
    path = a_star_search(env_map, agent_position, target_frontier, action_set)
    print(f"path: {path}")
    new_agent_position = path[-1]

    # Convert the path to an action sequence
    action_sequence = convert_path_to_actions(path, action_set)

    return action_sequence, new_agent_position
    

def a_star_search(env_map, start, goal, action_set):
    """
    A* search algorithm for pathfinding.

    Args:
    env_map (numpy.array): The map of the environment.
    start (tuple): The starting position (row, col).
    goal (tuple): The goal position (row, col).
    action_set (list of tuples): The set of possible actions.

    Returns:
    list of tuples: The path from start to goal as a sequence of positions.
    """
    # Helper function to calculate heuristic (Euclidean distance)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Initialize open and closed sets
    open_set = {start}
    came_from = {}

    # Cost from start along the best known path
    g_score = {start: 0}

    # Estimated total cost from start to goal through y
    f_score = {start: heuristic(start, goal)}

    while open_set:
        # Get the node in open_set having the lowest f_score[] value
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path
        print(f"open set: {open_set}")
        open_set.remove(current)
        print(f"open set: {open_set}")
        for dx, dy in action_set:
            print(f"dx: {dx}, dy: {dy} of action set")
            neighbor = (current[0] + dx, current[1] + dy)
            print(f"neighbor: {neighbor}")
            # Check if within map bounds and not an obstacle
            print(f"env_map[neighbor] = {env_map[neighbor]}")
            if 0 <= neighbor[0] < env_map.shape[0] and 0 <= neighbor[1] < env_map.shape[1] and env_map[neighbor] != -1:
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                print(f"tentative_g_score: {tentative_g_score}")
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.add(neighbor)
                    print(f"open set: {open_set}")

    # Path not found
    return []

def convert_path_to_actions(path, action_set):
    """
    Convert a path of positions to a sequence of actions.

    Args:
    path (list of tuples): The path as a sequence of positions.
    action_set (list of tuples): The set of possible actions.

    Returns:
    list of tuples: The path as a sequence of actions.
    """
    action_sequence = []
    for i in range(len(path) - 1):
        action = tuple(np.subtract(path[i + 1], path[i]))
        if action in action_set:
            action_sequence.append(action)
        else:
            # Handle case where action is not in action set (should not happen in this scenario)
            action_sequence.append(None)

    return action_sequence

def frontier_exploration(observation_tensor, position):
    """
    Perform the frontier exploration process.

    Args:
    observation_tensor (numpy.array): A NxNx3 tensor representing the partial observation.

    Returns:
    tuple: The new position of the agent after completing the exploration step.
    """
    # Initialize the environment
    env = initialize_environment(observation_tensor, position)
    env_map = env['map']
    agent_position = env['agent_position']

    action_seq = list()
    new_pos = tuple()

    while True:
        # Identify frontiers
        frontiers = identify_frontiers(env_map)
        print(frontiers)

        # If no frontiers are found, exploration is complete
        if not frontiers:
            break

        # Find the nearest frontier
        target_frontier = find_nearest_frontier(frontiers, agent_position)
        print(f"target frontier: {target_frontier}")

        # Plan a path to the nearest frontier
        action_seq, new_pos = plan_path_to_frontier(
            env_map, 
            agent_position, 
            target_frontier, 
            [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        )
        print(f"action seq: {action_seq}")

        # In a real implementation, here we would update the observation tensor
        # based on new observations made by the agent after moving

    return action_seq, new_pos

# This function encapsulates the entire frontier exploration process.

