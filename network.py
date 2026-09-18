import networkx as nx
import numpy as np

from config import (
    SENSOR_COMMUNICATION_RANGE,
    UAV_COMMUNICATION_RANGE
)


class DynamicWSNGraph:

    def __init__(self):

        self.graph = nx.Graph()


    def calculate_distance(self, position1, position2):

        return np.linalg.norm(
            np.array(position1) - np.array(position2)
        )


    def build_graph(self, environment):

        self.graph = nx.Graph()

        # Add sensor nodes

        for sensor in environment.sensors:

            self.graph.add_node(
                f"S{sensor.node_id}",
                node_type="sensor",
                position=sensor.position,
                energy=sensor.energy,
                priority=sensor.priority
            )


        # Add UAV node

        self.graph.add_node(
            "UAV",
            node_type="uav",
            position=environment.uav.position,
            energy=environment.uav.energy
        )


        # Add command station

        self.graph.add_node(
            "CS",
            node_type="command_station",
            position=environment.command_station.position
        )


        # Sensor-to-sensor connections

        for i in range(len(environment.sensors)):

            for j in range(i + 1, len(environment.sensors)):

                sensor1 = environment.sensors[i]
                sensor2 = environment.sensors[j]

                distance = self.calculate_distance(
                    sensor1.position,
                    sensor2.position
                )

                if distance <= SENSOR_COMMUNICATION_RANGE:

                    link_quality = (
                        1 - distance / SENSOR_COMMUNICATION_RANGE
                    )

                    self.graph.add_edge(
                        f"S{sensor1.node_id}",
                        f"S{sensor2.node_id}",
                        distance=distance,
                        link_quality=link_quality
                    )


        # Sensor-to-UAV connections

        for sensor in environment.sensors:

            distance = self.calculate_distance(
                sensor.position,
                environment.uav.position
            )

            if distance <= UAV_COMMUNICATION_RANGE:

                link_quality = (
                    1 - distance / UAV_COMMUNICATION_RANGE
                )

                self.graph.add_edge(
                    f"S{sensor.node_id}",
                    "UAV",
                    distance=distance,
                    link_quality=link_quality
                )


        # UAV-to-Command Station connection

        distance = self.calculate_distance(
            environment.uav.position,
            environment.command_station.position
        )

        if distance <= UAV_COMMUNICATION_RANGE:

            link_quality = (
                1 - distance / UAV_COMMUNICATION_RANGE
            )

            self.graph.add_edge(
                "UAV",
                "CS",
                distance=distance,
                link_quality=link_quality
            )


        return self.graph