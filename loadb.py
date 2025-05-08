import random
import time

# Simulate a server by keeping track of active connections
class Server:
    def __init__(self, name):
        self.name = name
        self.active_connections = 0

    def handle_request(self):
        # Simulate handling a request
        self.active_connections += 1
        time.sleep(random.uniform(0.5, 1.5))  # Simulate work
        self.active_connections -= 1

# Function to simulate Round Robin load balancing
def round_robin(servers, index):
    return servers[index % len(servers)]

# Function to simulate Least Connections load balancing
def least_connections(servers):
    return min(servers, key=lambda server: server.active_connections)

# Simulate client requests and distribute them using load balancing
def simulate_requests(servers, algorithm="round_robin"):
    for client_id in range(1, 6):  # Simulate 5 clients
        if algorithm == "round_robin":
            server = round_robin(servers, client_id - 1)
        elif algorithm == "least_connections":
            server = least_connections(servers)
        
        print(f"Client {client_id} routed to {server.name}")
        server.handle_request()

# Main function to simulate the load balancing
def main():
    # Create 3 servers
    servers = [Server(f"Server {i+1}") for i in range(3)]

    # Simulate requests using Round Robin
    print("\n--- Round Robin Load Balancing ---")
    simulate_requests(servers, "round_robin")

    # Simulate requests using Least Connections
    print("\n--- Least Connections Load Balancing ---")
    simulate_requests(servers, "least_connections")

if __name__ == "__main__":
    main()
